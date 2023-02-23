
# Import needed dependancies

from time import gmtime
from time import strftime
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pelicun.base import convert_to_MultiIndex
from pelicun.base import convert_to_SimpleIndex
from pelicun.base import describe
from pelicun.base import EDP_to_demand_type
from pelicun.file_io import load_data
from pelicun.assessment import Assessment

class LossAssesment:
    
    def __init__(self,stripe_num,config_path):
        
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        self.Demand=pd.DataFrame(data=None, columns=['a'])
        self.stripe_num = stripe_num
        self.additional_fragility_db = pd.DataFrame(data=None, columns=['a'])
        self.configPath = config_path
        # self._results
        
    def get_input(self):
        
        self.log_msg('First line of DL_calculation')
        
        self.loadConfig(self.configPath)
        self.initializePelicun(self.options)
        self.getDemand()
        # self.removeCollapses()
        self.generateDemandSample()
        self.loadDemandSample()
        self.saveDemandSampleRes()
        
        self.getComponentsList()
        self.loadCmpSample()
        self.getComponentsFrag()
        self.addAdittionalFragConsq()
        self.loadDamageModel()
        self.loadConseq()
        return self
        
    def run_assesment(self):
        
        self.calcDamage()
        self.getLossMap()
        self.calcLosses()
        return self
    
    def processResults(self):
        
        self.getLossSummary()
        
        
        
    @ classmethod 
    
    def initializePelicun(self,options):
        self.PAL = Assessment(options)
    
    def loadConfig(self,config_path):
        config_path = Path(config_path).resolve()

        # open the file and load the contents
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        DL_config = config.get('DL', None)
        
        if DL_config is None:

            self.log_msg("Damage and Loss configuration missing from input file. "
                    "Terminating analysis.")
            return -1

        self.asset_config = DL_config.get('Asset', None)
        self.demand_config = DL_config.get('Demands', None)
        self.damage_config = DL_config.get('Damage', None)
        self.loss_config = DL_config.get('Losses', None)
        self.out_config = config.get('out_config', None)
        
        self.options = DL_config.get("Options", {})
        self.options.update({
            "LogFile": "pelicun_log.txt",
            "Verbose": True
            })


        if self.asset_config is None:
            self.log_msg("Asset configuration missing. Terminating analysis.")
            return -1

        if self.demand_config is None:
            self.log_msg("Demand configuration missing. Terminating analysis.")
            return -1

        # get the length unit from the config file
        try:
            self.length_unit = config['GeneralInformation']['units']['length']
        except KeyError:
            self.log_msg(
                "No default length unit provided in the input file. "
                "Terminating analysis. ")

            return -1

        if self.out_config is None:
            self.log_msg("Output configuration missing. Terminating analysis.")
            return -1

        # initialize the Pelicun Assessement
        options = DL_config.get("Options", {})
        options.update({
            "LogFile": "pelicun_log.txt",
            "Verbose": True
            })
        
    def getDemand(self):  
        
        self.demand_path = Path(self.demand_config['DemandFilePath']).resolve()
        raw_demands = pd.read_csv(self.demand_path, index_col=0)
        
        # add units to the demand data if needed
        if "Units" not in raw_demands.index:

            demands = self.add_units(raw_demands, self.length_unit)

        else:
            demands = raw_demands
            
        self.demands = demands
        
    def removeCollapses(self):
        
        
        # remove excessive demands that are considered collapses, if needed
        if self.demand_config.get('CollapseLimits', False):

            raw_demands = convert_to_MultiIndex(self.demands, axis=1)

            DEM_to_drop = np.full(raw_demands.shape[0], False)
            
            idx = pd.IndexSlice

            for DEM_type, limit in self.demand_config['CollapseLimits'].items():

                if raw_demands.columns.nlevels == 4:
                    DEM_to_drop += raw_demands.loc[
                                   :, idx[:, DEM_type, :, :]].max(axis=1) > float(limit)

                else:
                    DEM_to_drop += raw_demands.loc[
                                   :, idx[DEM_type, :, :]].max(axis=1) > float(limit)

            self.demands = raw_demands.loc[~DEM_to_drop, :]

            self.log_msg(f"{np.sum(DEM_to_drop)} realizations removed from the demand "
                    f"input because they exceed the collapse limit. The remaining "
                    f"sample size: {raw_demands.shape[0]}")
    
    def getComponentsList(self):
        
        if self.asset_config.get('ComponentAssignmentFile', False):
            cmp_marginals = pd.read_csv(self.asset_config['ComponentAssignmentFile'],
                                        index_col=0, encoding_errors='replace')   
            
        self.cmp_marginals = cmp_marginals
        
    def getComponentsFrag(self):
        
        default_DBs = {
            'fragility': {
                'FEMA P-58': 'fragility_DB_FEMA_P58_2nd.csv',
                'Hazus Earthquake': 'fragility_DB_HAZUS_EQ.csv'
            },
            'repair': {
                'FEMA P-58': 'bldg_repair_DB_FEMA_P58_2nd.csv',
                'Hazus Earthquake': 'bldg_repair_DB_HAZUS_EQ.csv'
            }
        
        }
        
        if self.damage_config is not None:

            # load the fragility information
            if self.asset_config['ComponentDatabase'] != "User Defined":
                fragility_db = (
                    'PelicunDefault/' +
                    default_DBs['fragility'][self.asset_config['ComponentDatabase']])

            else:
                fragility_db = self.asset_config['ComponentDatabasePath']
                
            self.fragility_db=fragility_db
    
    def addAdittionalFragConsq(self):
        
        cmp_list=self.cmp_marginals.index.unique().values[:-3]
        
        # add missing data to P58 damage model
        P58_data = self.PAL.get_default_data('fragility_DB_FEMA_P58_2nd')
        # P58_data.to_excel(r'P58_data.xlsx', index=False)
        
        # now take those components that are incomplete, and add the missing information
        additional_fragility_db = P58_data.loc[cmp_list,:].loc[P58_data.loc[cmp_list,'Incomplete'] == 1].sort_index()
        
        # D3052.013i - Air Handling Unit
        # use a placeholder of 3.0|0.5
        additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_0')] = 3.0
        additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_1')] = 0.5
        
        self.adf = additional_fragility_db
        
        # TODO: COLLAPSE FRAGILITY 
   
        
    def loadDamageModel(self):
        
            self.PAL.damage.load_damage_model([self.fragility_db,self.adf])

            # load the damage process if needed
            dmg_process = None
            if self.damage_config.get('DamageProcess', False):
                
                damage_processes = {
                    'FEMA P-58': {
                        "1_collapse": {
                            "DS1": "ALL_NA"
                        },
                        "2_excessiveRID": {
                            "DS1": "irreparable_DS1"
                        }
                    },

                    'Hazus Earthquake': {
                        "1_STR": {
                            "DS5": "collapse_DS1"
                        },
                        "2_LF": {
                            "DS5": "collapse_DS1"
                        },
                        "3_collapse": {
                            "DS1": "ALL_NA"
                        },
                        "4_excessiveRID": {
                            "DS1": "irreparable_DS1"
                        }
                    }
                }

                dp_approach = self.damage_config['DamageProcess']

                if dp_approach in damage_processes:
                    dmg_process = damage_processes[dp_approach]


                elif dp_approach == "User Defined":

                    # load the damage process from a file
                    with open(self.damage_config['DamageProcessFilePath'], 'r',
                              encoding='utf-8') as f:
                        dmg_process = json.load(f)

                else:
                    self.log_msg(f"Prescribed Damage Process not recognized: "
                            f"{dp_approach}")
                self.dmg_process = dmg_process
                    
    def calcDamage(self):
        
        self.PAL.damage.calculate(dmg_process=self.dmg_process)
                        
        damage_sample = self.PAL.damage.save_sample()
        self.damage_sample = damage_sample
        
        # if requested, save results
        if self.out_config.get('Damage', False):

            out_reqs = [out if val else ""
                        for out, val in self.out_config['Damage'].items()]

            if np.any(np.isin(['Sample', 'Statistics',
                               'GroupedSample', 'GroupedStatistics'],
                              out_reqs)):


                if 'Sample' in out_reqs:
                    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)
                    damage_sample_s.to_csv(
                        "DMG_sample.zip",
                        index_label=damage_sample_s.columns.name,
                        compression=dict(method='zip',
                                         archive_name='DMG_sample.csv'))

                if 'Statistics' in out_reqs:
                    damage_stats = convert_to_SimpleIndex(describe(damage_sample),
                                                          axis=1)
                    damage_stats.to_csv("DMG_stats.csv",
                                        index_label=damage_stats.columns.name)

                if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                    grp_damage = damage_sample.groupby(level=[0, 3], axis=1).sum()

                    if 'GroupedSample' in out_reqs:
                        grp_damage_s = convert_to_SimpleIndex(grp_damage, axis=1)
                        grp_damage_s.to_csv("DMG_grp.zip",
                                            index_label=grp_damage_s.columns.name,
                                            compression=dict(
                                                method='zip',
                                                archive_name='DMG_grp.csv'))

                    if 'GroupedStatistics' in out_reqs:
                        grp_stats = convert_to_SimpleIndex(describe(grp_damage),
                                                           axis=1)
                        grp_stats.to_csv("DMG_grp_stats.csv",
                                         index_label=grp_stats.columns.name)
                     
    def loadConseq(self): 
        
        default_DBs = {
            'fragility': {
                'FEMA P-58': 'fragility_DB_FEMA_P58_2nd.csv',
                'Hazus Earthquake': 'fragility_DB_HAZUS_EQ.csv'
            },
            'repair': {
                'FEMA P-58': 'bldg_repair_DB_FEMA_P58_2nd.csv',
                'Hazus Earthquake': 'bldg_repair_DB_HAZUS_EQ.csv'
            }

        }
        
    # if a loss assessment is requested
        if self.loss_config is not None:

          out_config_loss = self.out_config.get('Loss', {})

          # if requested, calculate repair consequences
          if self.loss_config.get('BldgRepair', False):

              bldg_repair_config = self.loss_config['BldgRepair']

              # load the consequence information
              if bldg_repair_config['ConsequenceDatabase'] != "User Defined":
                  consequence_db = (
                          'PelicunDefault/' +
                          default_DBs['repair'][
                              bldg_repair_config['ConsequenceDatabase']])

                  conseq_df = self.PAL.get_default_data(consequence_db[:-4])

              else:
                  consequence_db = Path(bldg_repair_config['ConsequenceDatabaseFile']).resolve()
                  conseq_df = load_data(consequence_db,1,
                      orientation=1, reindex=False, convert=[])

              # add the replacement consequence to the data
              adf = pd.DataFrame(
                  columns=conseq_df.columns,
                  index=pd.MultiIndex.from_tuples(
                      [('replacement', 'Cost'), ('replacement', 'Time')]))

              DL_method = bldg_repair_config['ConsequenceDatabase']
              rc = ('replacement', 'Cost')
              if 'ReplacementCost' in bldg_repair_config.keys():
                  rCost_config = bldg_repair_config['ReplacementCost']

                  adf.loc[rc, ('Quantity', 'Unit')] = "1 EA"

                  adf.loc[rc, ('DV', 'Unit')] = rCost_config["Unit"]

                  adf.loc[rc, ('DS1', 'Theta_0')] = rCost_config["Median"]

                  if rCost_config.get('Distribution', 'N/A') != 'N/A':
                      adf.loc[rc, ('DS1', 'Family')] = rCost_config[
                          "Distribution"]
                      adf.loc[rc, ('DS1', 'Theta_1')] = rCost_config[
                          "Theta_1"]

              else:
                  # add a default replacement cost value as a placeholder
                  # the default value depends on the consequence database

                  # for FEMA P-58, use 0 USD
                  if DL_method == 'FEMA P-58':
                      adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rc, ('DV', 'Unit')] = 'USD_2011'
                      adf.loc[rc, ('DS1', 'Theta_0')] = 0

                  # for Hazus EQ, use 1.0 as a loss_ratio
                  elif DL_method == 'Hazus Earthquake':
                      adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'

                      # store the replacement cost that corresponds to total loss
                      adf.loc[rc, ('DS1', 'Theta_0')] = 100.0

                  # otherwise, use 1 (and expect to have it defined by the user)
                  else:
                      adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'
                      adf.loc[rc, ('DS1', 'Theta_0')] = 1

              rt = ('replacement', 'Time')
              if 'ReplacementTime' in bldg_repair_config.keys():
                  rTime_config = bldg_repair_config['ReplacementTime']
                  rt = ('replacement', 'Time')

                  adf.loc[rt, ('Quantity', 'Unit')] = "1 EA"

                  adf.loc[rt, ('DV', 'Unit')] = rTime_config["Unit"]

                  adf.loc[rt, ('DS1', 'Theta_0')] = rTime_config["Median"]

                  if rTime_config.get('Distribution', 'N/A') != 'N/A':
                      adf.loc[rt, ('DS1', 'Family')] = rTime_config[
                          "Distribution"]
                      adf.loc[rt, ('DS1', 'Theta_1')] = rTime_config[
                          "Theta_1"]
              else:
                  # add a default replacement time value as a placeholder
                  # the default value depends on the consequence database

                  # for FEMA P-58, use 0 worker_days
                  if DL_method == 'FEMA P-58':
                      adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rt, ('DV', 'Unit')] = 'worker_day'
                      adf.loc[rt, ('DS1', 'Theta_0')] = 0

                  # for Hazus EQ, use 1.0 as a loss_ratio
                  elif DL_method == 'Hazus Earthquake':
                      adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rt, ('DV', 'Unit')] = 'day'

                      # load the replacement time that corresponds to total loss
                      occ_type = self.asset_config['OccupancyType']
                      adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
                          (f"STR.{occ_type}", 'Time'), ('DS5', 'Theta_0')]

                  # otherwise, use 1 (and expect to have it defined by the user)
                  else:
                      adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                      adf.loc[rt, ('DV', 'Unit')] = 'loss_ratio'
                      adf.loc[rt, ('DS1', 'Theta_0')] = 1
                      
              self.loss_cmps = conseq_df.index.unique(level=0)        
              self.conseq_df = [conseq_df, adf]


    def generateDemandSample(self):
        
        # get the calibration information
        if self.demand_config.get('Calibration', False):

            # load the available demand sample
            self.PAL.demand.load_sample(self.demands)

            # then use it to calibrate the demand model
            self.PAL.demand.calibrate_model(self.demand_config['Calibration'])

            # and generate a new demand sample
            sample_size = int(self.demand_config['SampleSize'])

            self.PAL.demand.generate_sample({"SampleSize": sample_size})

        # get the generated demand sample
        demand_sample, demand_units = self.PAL.demand.save_sample(save_units=True)

        self.demand_sample = pd.concat([demand_sample, demand_units.to_frame().T])
        
        # add a constant zero demand
        demand_sample[('ONE', '0', '1')] = np.ones(demand_sample.shape[0])
        demand_sample.loc['Units', ('ONE', '0', '1')] = 'ea'
        
        self.demand_sample = demand_sample

        
    def loadDemandSample(self):
        self.PAL.demand.load_sample(convert_to_SimpleIndex(self.demand_sample, axis=1))


    def saveDemandSampleRes(self):
        if self.out_config.get('Demand', False):

            out_reqs = [out if val else "" for out, val in self.out_config['Demand'].items()]

            if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
                demand_sample = self.PAL.demand.save_sample()

                if 'Sample' in out_reqs:
                    demand_sample_s = convert_to_SimpleIndex(demand_sample, axis=1)
                    demand_sample_s.to_csv("DEM_sample.zip",
                                           index_label=demand_sample_s.columns.name,
                                           compression=dict(
                                               method='zip',
                                               archive_name='DEM_sample.csv'))

                if 'Statistics' in out_reqs:
                    demand_stats = convert_to_SimpleIndex(
                        describe(demand_sample), axis=1)
                    demand_stats.to_csv("DEM_stats.csv",
                                        index_label=demand_stats.columns.name)
                    
    def loadCmpSample(self):
        
        # set the number of stories
        if self.asset_config.get('NumberOfStories', False):
           self.PAL.stories = int(self.asset_config['NumberOfStories'])
        
        # load a component model and generate a sample
        if self.asset_config.get('ComponentAssignmentFile', False):

            cmp_marginals = pd.read_csv(self.asset_config['ComponentAssignmentFile'],
                                        index_col=0, encoding_errors='replace')

            # add a component to support collapse calculation
            cmp_marginals.loc['collapse', 'Units'] = 'ea'
            cmp_marginals.loc['collapse', 'Location'] = 0
            cmp_marginals.loc['collapse', 'Direction'] = 1
            cmp_marginals.loc['collapse', 'Theta_0'] = 1.0

            # add components to support irreparable damage calculation
            if 'IrreparableDamage' in self.damage_config.keys():

                DEM_types = self.demand_sample.columns.unique(level=0)
                if 'RID' in DEM_types:

                    # excessive RID is added on every floor to detect large RIDs
                    cmp_marginals.loc['excessiveRID', 'Units'] = 'ea'

                    locs = self.demand_sample['RID'].columns.unique(level=0)
                    cmp_marginals.loc['excessiveRID', 'Location'] = ','.join(locs)

                    dirs = self.demand_sample['RID'].columns.unique(level=1)
                    cmp_marginals.loc['excessiveRID', 'Direction'] = ','.join(dirs)

                    cmp_marginals.loc['excessiveRID', 'Theta_0'] = 1.0

                    # irreparable is a global component to recognize is any of the
                    # excessive RIDs were triggered
                    cmp_marginals.loc['irreparable', 'Units'] = 'ea'
                    cmp_marginals.loc['irreparable', 'Location'] = 0
                    cmp_marginals.loc['irreparable', 'Direction'] = 1
                    cmp_marginals.loc['irreparable', 'Theta_0'] = 1.0

                else:
                    self.log_msg('WARNING: No residual interstory drift ratio among'
                            'available demands. Irreparable damage cannot be '
                            'evaluated.')

            # load component model
            self.PAL.asset.load_cmp_model({'marginals': self.cmp_marginals})

            # generate component quantity sample
            self.PAL.asset.generate_cmp_sample()

        # if requested, load the quantity sample from a file
        elif self.asset_config.get('ComponentSampleFile', False):
            self.PAL.asset.load_cmp_sample(self.asset_config['ComponentSampleFile'])

        self.cmp_sample = self.PAL.asset.save_cmp_sample()
        
    def getLossMap(self):
        
        # prepare the loss map
        
        bldg_repair_config=self.loss_config['BldgRepair']
        loss_map = None
        if bldg_repair_config['MapApproach'] == "Automatic":

            # get the damage sample
            dmg_sample = self.PAL.damage.save_sample()

            # create a mapping for all components that are also in
            # the prescribed consequence database
            dmg_cmps = dmg_sample.columns.unique(level='cmp')
            loss_cmps = self.loss_cmps

            drivers = []
            loss_models = []
            
            DL_method = bldg_repair_config['ConsequenceDatabase']

            if DL_method == 'FEMA P-58':

                # with FEMA P-58 we assume fragility and consequence data
                # have the same IDs

                for dmg_cmp in dmg_cmps:

                    if dmg_cmp == 'collapse':
                        continue

                    if dmg_cmp in loss_cmps:
                        drivers.append(f'DMG-{dmg_cmp}')
                        loss_models.append(dmg_cmp)

            elif DL_method == 'Hazus Earthquake':

                # with Hazus Earthquake we assume that consequence
                # archetypes are only differentiated by occupancy type
                occ_type = self.asset_config['OccupancyType']

                for dmg_cmp in dmg_cmps:

                    if dmg_cmp == 'collapse':
                        continue

                    cmp_class = dmg_cmp.split('.')[0]
                    loss_cmp = f'{cmp_class}.{occ_type}'

                    if loss_cmp in loss_cmps:
                        drivers.append(f'DMG-{dmg_cmp}')
                        loss_models.append(loss_cmp)


            elif DL_method == 'User Defined':
                
                # With User Defined Consequence database
                # get loss mapping from user defined file
                
                # MapFilePath = Path(bldg_repair_config['MapFile']).resolve()
                # loss_map = pd.read_csv(MapFilePath,
                #                        index_col=0)
                
                # Here, with User Defined consequence database
                # assume the frgilities and consequences func hav the same name
                for dmg_cmp in dmg_cmps:
    
                    if dmg_cmp == 'collapse':
                        continue
    
                    if dmg_cmp in loss_cmps:
                        drivers.append(f'DMG-{dmg_cmp}')
                        loss_models.append(dmg_cmp)
                        
            loss_map = pd.DataFrame(loss_models,
                                    columns=['BldgRepair'],
                                    index=drivers)

            # prepare additional loss map entries, if needed
            if 'DMG-collapse' not in loss_map.index:
                loss_map.loc['DMG-collapse',    'BldgRepair'] = 'replacement'
                loss_map.loc['DMG-irreparable', 'BldgRepair'] = 'replacement'

        self.loss_map=loss_map
        
    def calcLosses(self):
        
        self.PAL.bldg_repair.load_model(self.conseq_df, self.loss_map)
        self.PAL.bldg_repair.calculate()
        
 
    def getLossSummary(self):
        
       self.agg_repair = self.PAL.bldg_repair.aggregate_losses()

       damage_sample = self.damage_sample.groupby(level=[0, 3], axis=1).sum()
       damage_sample_s = convert_to_SimpleIndex(self.damage_sample, axis=1)
        
       if 'collapse-1' in damage_sample_s.columns:
            damage_sample_s['collapse'] = damage_sample_s['collapse-1']
       else:
            damage_sample_s['collapse'] = np.zeros(damage_sample_s.shape[0])
        
       if 'irreparable-1' in damage_sample_s.columns:
            damage_sample_s['irreparable'] = damage_sample_s['irreparable-1']
       else:
            damage_sample_s['irreparable'] = np.zeros(damage_sample_s.shape[0])
        
       agg_repair_s = convert_to_SimpleIndex(self.agg_repair, axis=1)
        
       summary = pd.concat([agg_repair_s,
                             damage_sample_s[['collapse', 'irreparable']]],
                            axis=1)
        
       summary_stats = describe(summary)
        
        # save summary sample
       summary.to_csv("DL_summary.csv", index_label='#')
        
        # save summary statistics
       summary_stats.to_csv("DL_summary_stats.csv")
       
       # self.results.repairCost=summary_stats.loc[:,"repair_cost-"]
        
       return 0
       
       
       
    @ staticmethod    
    def log_msg(msg):

        formatted_msg = f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} {msg}'
    
        print(formatted_msg)
        
    @ staticmethod 
    def add_units(raw_demands, length_unit):
    
        demands = raw_demands.T
    
        demands.insert(0, "Units", np.nan)
    
        if length_unit == 'in':
            length_unit = 'inch'
    
        demands = convert_to_MultiIndex(demands, axis=0).sort_index(axis=0).T
    
        if demands.columns.nlevels == 4:
            DEM_level = 1
        else:
            DEM_level = 0
    
        # drop demands with no EDP type identified
        demands.drop(demands.columns[
            demands.columns.get_level_values(DEM_level) == ''],
                     axis=1, inplace=True)
    
        # assign units
        demand_cols = demands.columns.get_level_values(DEM_level)
    
        # remove additional info from demand names
        demand_cols = [d.split('_')[0] for d in demand_cols]
    
        # acceleration
        acc_EDPs = ['PFA', 'PGA', 'SA']
        EDP_mask = np.isin(demand_cols, acc_EDPs)
    
        if np.any(EDP_mask):
            demands.iloc[0, EDP_mask] = length_unit+'ps2'
    
        # speed
        speed_EDPs = ['PFV', 'PWS', 'PGV', 'SV']
        EDP_mask = np.isin(demand_cols, speed_EDPs)
    
        if np.any(EDP_mask):
            demands.iloc[0, EDP_mask] = length_unit+'ps'
    
        # displacement
        disp_EDPs = ['PFD', 'PIH', 'SD', 'PGD']
        EDP_mask = np.isin(demand_cols, disp_EDPs)
    
        if np.any(EDP_mask):
            demands.iloc[0, EDP_mask] = length_unit
    
        # rotation
        rot_EDPs = ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']
        EDP_mask = np.isin(demand_cols, rot_EDPs)
    
        if np.any(EDP_mask):
            demands.iloc[0, EDP_mask] = 'rad'
    
        # convert back to simple header and return the DF
        return convert_to_SimpleIndex(demands, axis=1)


#%% EXECUTION


Ass = LossAssesment(7,r'C:\Users\dsuas\My_Drive\UME\3.Codes\LossAssesment\input.json')
Ass = Ass.get_input()
Ass = Ass.run_assesment()
Ass = Ass.processResults()


