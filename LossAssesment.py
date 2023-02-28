
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
import matplotlib.pyplot as plt
from pelicun.base import convert_to_MultiIndex
from pelicun.base import convert_to_SimpleIndex
from pelicun.base import describe
from pelicun.base import EDP_to_demand_type
from pelicun.file_io import load_data
from pelicun.assessment import Assessment


from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default='svg'

class LossAssesment:
    
    def __init__(self,stripe,config_path):
        
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        self.Demand=pd.DataFrame(data=None, columns=['a'])
        self.stripe = stripe
        self.additional_fragility_db = pd.DataFrame(data=None, columns=['a'])
        self.configPath = config_path
        # self._results
        
    def get_input(self):
        
        self.log_msg('First line of DL_calculation')
        
        self.loadConfig(self.configPath)
        self.initializePelicun(self.options)
        
        self.getDemandFromAnalysisSample()
        self.generateDemandSample()
        self.getRID()
        self.loadDemandSample()

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
        
        self.getOutFiles()
        return self
        
        
    @ classmethod 
      
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
        
    def initializePelicun(self,options):
        self.PAL = Assessment(options)
           
    def getDemandFromAnalysisSample(self):  
        
        # check if there is a demand file location specified in the config file
        if self.demand_config.get('DemandFilePath', False):
    
            self.demand_path = Path(self.demand_config['DemandFilePath']).resolve()
    
        else:
            # otherwise assume that there is a response.csv file next to the config file
            self.demand_path = self.config_path.parent/'response.csv'
        
        raw_demands = pd.read_csv(self.demand_path, index_col=0)
        
        # slice the demand to get the stripe demand only   
        raw_demands = raw_demands[raw_demands.columns
                                  [pd.Series(raw_demands.columns)
                                   .str.startswith(self.stripe)]]
        
        raw_demands = convert_to_MultiIndex(raw_demands, axis=1)
        
        
        # remove excessive demands that are considered collapses, if needed
        if self.demand_config.get('CollapseLimits', False):
            
            self.removeCollapses(raw_demands)
        

        
        # add units to the demand data if needed
        if "Units" not in raw_demands.index:

            demands = self.add_units(self.demands, self.length_unit)

        else:
            demands = raw_demands
            
        self.demands = demands
        
        def  getDemandFromAnalysisStats(self): 
            
            # gets the demand from a csv file specifing the mean and log std dev
            # for each stripe 
            
            # check if there is a demand file location specified in the config file
            if self.demand_config.get('DemandFilePath', False):
        
                self.demand_path = Path(self.demand_config['DemandFilePath']).resolve()
        
            else:
                # otherwise assume that there is a response.csv file next to the config file
                self.demand_path = self.config_path.parent/'response.csv'
            
            raw_demands = pd.read_csv(self.demand_path, index_col=0)
            
            # slice the demand to get the stripe demand only   
            raw_demands = raw_demands[raw_demands.columns
                                      [pd.Series(raw_demands.columns)
                                       .str.startswith(self.stripe)]]
            
            raw_demands = convert_to_MultiIndex(raw_demands, axis=1)
            
            # units - - - - - - - - - - - - - - - - - - - - - - - -  
            raw_demands.insert(0, 'Units',"")
            
            # PFA is in "g" in this example, while PID is "rad"
            raw_demands.loc['PFA','Units'] = 'g'
            raw_demands.loc['PID','Units'] = 'rad'
            
            # distribution family  - - - - - - - - - - - - - - - - -  
            raw_demands.insert(1, 'Family',"")
            
            # we assume lognormal distribution for all demand marginals
            raw_demands['Family'] = 'lognormal'
            
            # distribution parameters  - - - - - - - - - - - - - - -
            # pelicun uses generic parameter names to handle various distributions within the same data structure
            # we need to rename the parameter columns as follows:
            # median -> theta_0
            # log_std -> theta_1
            raw_demands.rename(columns = {'mu': 'Theta_0'}, inplace=True)
            raw_demands.rename(columns = {'log_std': 'Theta_1'}, inplace=True)

            self.demands = raw_demands
            
        
    def removeCollapses(self, raw_demands):
        

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
                f"sample size: {self.demands.shape[0]}")
        
        return self
    
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
        self.demand_units = demand_units
        
        # add a constant zero demand
        demand_sample[('ONE', '0', '1')] = np.ones(demand_sample.shape[0])
        demand_sample.loc['Units', ('ONE', '0', '1')] = 'ea'
        
        self.demand_sample = demand_sample
       
    def getRID(self):
        
        idx = pd.IndexSlice
        
        # get residual drift estimates, if needed
        if self.demand_config.get('InferResidualDrift', False):
    
            RID_config = self.demand_config['InferResidualDrift']
    
            if RID_config['method'] == 'FEMA P-58':
    
                RID_list = []
                PID = self.demand_sample['PID'].copy()
                PID.drop('Units', inplace=True)
                PID = PID.astype(float)
    
                for direction, delta_yield in RID_config.items():
    
                    if direction == 'method':
                        continue
    
                    RID = self.PAL.demand.estimate_RID(
                        PID.loc[:, idx[:, direction]],
                        {'yield_drift': float(delta_yield)})
    
                    RID_list.append(RID)
    
                RID = pd.concat(RID_list, axis=1)
                RID_units = pd.Series(['rad', ]*RID.shape[1], index=RID.columns,
                                      name='Units')
            self.RID = RID
            self.RID_sample = pd.concat([RID, RID_units.to_frame().T])
            
            self.demand_sample = pd.concat([self.demand_sample, self.RID_sample], axis=1)
    
    def loadDemandSample(self):
        self.demand_sample = self.PAL.demand.save_sample()
          
    def getComponentsList(self):
        
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
                
            self.cmp_marginals = cmp_marginals
    
    def loadCmpSample(self):
        
        # set the number of stories
        if self.asset_config.get('NumberOfStories', False):
           self.PAL.stories = int(self.asset_config['NumberOfStories'])
        
        # load a component model and generate a sample
        if self.asset_config.get('ComponentAssignmentFile', False):
            # load component model
            self.PAL.asset.load_cmp_model({'marginals': self.cmp_marginals})

            # generate component quantity sample
            self.PAL.asset.generate_cmp_sample()

        # if requested, load the quantity sample from a file
        elif self.asset_config.get('ComponentSampleFile', False):
            self.PAL.asset.load_cmp_sample(self.asset_config['ComponentSampleFile'])

        self.cmp_sample = self.PAL.asset.save_cmp_sample()
         
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
        
        adf = pd.DataFrame(columns=P58_data.columns)

        if 'CollapseFragility' in self.damage_config.keys():

           coll_config = self.damage_config['CollapseFragility']

           adf.loc['collapse', ('Demand', 'Directional')] = 1
           adf.loc['collapse', ('Demand', 'Offset')] = 0

           coll_DEM = coll_config["DemandType"]

           if '_' in coll_DEM:
               coll_DEM, coll_DEM_spec = coll_DEM.split('_')
           else:
               coll_DEM_spec = None

           coll_DEM_name = None
           for demand_name, demand_short in EDP_to_demand_type.items():

               if demand_short == coll_DEM:
                   coll_DEM_name = demand_name
                   break

           if coll_DEM_name is None:
               return -1

           if coll_DEM_spec is None:
               adf.loc['collapse', ('Demand', 'Type')] = coll_DEM_name

           else:
               adf.loc['collapse', ('Demand', 'Type')] = \
                   f'{coll_DEM_name}|{coll_DEM_spec}'

           coll_DEM_unit = self.add_units(
               pd.DataFrame(columns=[f'{coll_DEM}-1-1', ]),
               self.length_unit).iloc[0, 0]

           adf.loc['collapse', ('Demand', 'Unit')] = coll_DEM_unit

           adf.loc['collapse', ('LS1', 'Family')] = (
               coll_config.get('CapacityDistribution', ""))

           adf.loc['collapse', ('LS1', 'Theta_0')] = (
               coll_config.get('CapacityMedian', ""))

           adf.loc['collapse', ('LS1', 'Theta_1')] = (
               coll_config.get('Theta_1', ""))

           adf.loc['collapse', 'Incomplete'] = 0

        else:

           # add a placeholder collapse fragility that will never trigger
           # collapse, but allow damage processes to work with collapse

           adf.loc['collapse', ('Demand', 'Directional')] = 1
           adf.loc['collapse', ('Demand', 'Offset')] = 0
           adf.loc['collapse', ('Demand', 'Type')] = 'One'
           adf.loc['collapse', ('Demand', 'Unit')] = 'ea'
           adf.loc['collapse', ('LS1', 'Theta_0')] = 2.0
           adf.loc['collapse', 'Incomplete'] = 0

        if 'IrreparableDamage' in self.damage_config.keys():

           irrep_config = self.damage_config['IrreparableDamage']

           # add excessive RID fragility according to settings provided in the
           # input file
           adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
           adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
           adf.loc['excessiveRID',
                   ('Demand', 'Type')] = 'Residual Interstory Drift Ratio'

           adf.loc['excessiveRID', ('Demand', 'Unit')] = 'rad'
           adf.loc['excessiveRID',
                   ('LS1', 'Theta_0')] = irrep_config['DriftCapacityMedian']

           adf.loc['excessiveRID',
                   ('LS1', 'Family')] = "lognormal"

           adf.loc['excessiveRID',
                   ('LS1', 'Theta_1')] = irrep_config['DriftCapacityLogStd']

           adf.loc['excessiveRID', 'Incomplete'] = 0

           # add a placeholder irreparable fragility that will never trigger
           # damage, but allow damage processes to aggregate excessiveRID here
           adf.loc['irreparable', ('Demand', 'Directional')] = 1
           adf.loc['irreparable', ('Demand', 'Offset')] = 0
           adf.loc['irreparable', ('Demand', 'Type')] = 'One'
           adf.loc['irreparable', ('Demand', 'Unit')] = 'ea'
           adf.loc['irreparable', ('LS1', 'Theta_0')] = 2.0
           adf.loc['irreparable', 'Incomplete'] = 0
        
        # now take those components that are incomplete, and add the missing information
        self.additional_fragility_db = P58_data.loc[cmp_list,:].loc[P58_data.loc[cmp_list,'Incomplete'] == 1].sort_index()
        
        # D3052.013i - Air Handling Unit
        # # use a placeholder of 3.0|0.5
        # additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_0')] = 3.0
        # additional_fragility_db.loc['D.30.52.013i',('LS1','Theta_1')] = 0.5
        
        self.adf = adf
          
    def loadDamageModel(self):
        
            self.PAL.damage.load_damage_model([self.fragility_db,
                                               self.additional_fragility_db,
                                               self.adf])

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

         
          # if requested, calculate repair consequences
          if self.loss_config.get('BldgRepair', False):

              bldg_repair_config = self.loss_config['BldgRepair']

              # load the consequence information
              if bldg_repair_config['ConsequenceDatabase'] != "User Defined":
                  consequence_db = (
                          'PelicunDefault/' +
                          default_DBs['repair'][
                              bldg_repair_config['ConsequenceDatabase']])

                  conseq_df = self.PAL.get_default_data(
                    default_DBs['repair'][
                        bldg_repair_config['ConsequenceDatabase']][:-4])


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
                  self.rCost_config=rCost_config

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
                      adf.loc[rc, ('DS1', 'Theta_0')] = 0

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
                      adf.loc[rt, ('DS1', 'Theta_0')] = 0
                      
              self.loss_cmps = conseq_df.index.unique(level=0)        
              self.conseq_df = conseq_df
              self.adf=adf
         
            
    def calcDamage(self):
        
        self.PAL.damage.calculate(dmg_process=self.dmg_process)             
        self.damage_sample = self.PAL.damage.save_sample()
                                      
    def getLossMap(self):
        
        # prepare the loss map
        
        bldg_repair_config=self.loss_config['BldgRepair']
        loss_map = None
        if bldg_repair_config['MapApproach'] == "Automatic":

            # get the damage sample
            self.dmg_sample = self.PAL.damage.save_sample()
            dmg_sample=self.dmg_sample

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
                # assume the frgilities and consequences func have the same name
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

        self.loss_map = loss_map
        self.loss_models = loss_models
        
    def calcLosses(self):
        
        self.PAL.bldg_repair.load_model([self.conseq_df,self.adf], self.loss_map)
        self.PAL.bldg_repair.calculate()
        self.repair_sample = self.PAL.bldg_repair.save_sample()
        self.agg_repair = self.PAL.bldg_repair.aggregate_losses()
        
    def getOutFiles(self):
        
        OutFolder = self.out_config.get('outFolder', False) 
         
                                
        if self.out_config.get('Demand', False):

            out_reqs = [out if val else "" for out, val in self.out_config['Demand'].items()]

            if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
                demand_sample = self.PAL.demand.save_sample()

                if 'Sample' in out_reqs:
                    demand_sample_s = convert_to_SimpleIndex(demand_sample, axis=1)
                    demand_sample_s.to_csv(os.path.join(OutFolder,"DEM_sample.zip"),
                                           index_label=demand_sample_s.columns.name,
                                           compression=dict(
                                               method='zip',
                                               archive_name='DEM_sample.csv'))

                if 'Statistics' in out_reqs:
                    demand_stats = convert_to_SimpleIndex(
                        describe(demand_sample), axis=1)
                    demand_stats.to_csv(os.path.join(OutFolder,"DEM_stats.csv"),
                                        index_label=demand_stats.columns.name)
                    
        # if requested, save results
        if self.out_config.get('Damage', False):
            
            damage_sample  = self.damage_sample
            
            out_reqs = [out if val else ""
                        for out, val in self.out_config['Damage'].items()]

            if np.any(np.isin(['Sample', 'Statistics',
                               'GroupedSample', 'GroupedStatistics'],
                              out_reqs)):


                if 'Sample' in out_reqs:
                    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)
                    damage_sample_s.to_csv( 
                        os.path.join(OutFolder,"DMG_sample.zip"),
                        index_label=damage_sample_s.columns.name,
                        compression=dict(method='zip',
                                         archive_name='DMG_sample.csv'))

                if 'Statistics' in out_reqs:
                    damage_stats = convert_to_SimpleIndex(describe(damage_sample),
                                                          axis=1)
                    damage_stats.to_csv(os.path.join(OutFolder,"DMG_stats.csv"),
                                        index_label=damage_stats.columns.name)

                if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                    grp_damage = damage_sample.groupby(level=[0, 3], axis=1).sum()

                    if 'GroupedSample' in out_reqs:
                        grp_damage_s = convert_to_SimpleIndex(grp_damage, axis=1)
                        grp_damage_s.to_csv(os.path.join(OutFolder,"DMG_grp.zip"),
                                            index_label=grp_damage_s.columns.name,
                                            compression=dict(
                                                method='zip',
                                                archive_name='DMG_grp.csv'))

                    if 'GroupedStatistics' in out_reqs:
                        grp_stats_damage = convert_to_SimpleIndex(describe(grp_damage),
                                                           axis=1)
                        grp_stats_damage.to_csv(os.path.join(OutFolder,"DMG_grp_stats.csv"),
                                         index_label=grp_stats_damage.columns.name)
                        
        if self.out_config.get('Loss', False):
            
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
            summary.to_csv(os.path.join(OutFolder,"DL_summary.csv"), index_label='#')
             
             # save summary statistics
            summary_stats.to_csv(os.path.join(OutFolder,"DL_summary_stats.csv"))
            
            # self.results.repairCost=summary_stats.loc[:,"repair_cost-"]
            
        
        out_config_loss = self.out_config.get('Loss', {})
        
        if out_config_loss.get('BldgRepair', False):

               out_reqs = [out if val else ""
                           for out, val in out_config_loss['BldgRepair'].items()]

               if np.any(np.isin(['Sample', 'Statistics',
                                  'GroupedSample', 'GroupedStatistics',
                                  'AggregateSample', 'AggregateStatistics'],
                                 out_reqs)):
                   
                   repair_sample = self.repair_sample
                   

                   if 'Sample' in out_reqs:
                       repair_sample_s = convert_to_SimpleIndex(
                           repair_sample, axis=1)
                       repair_sample_s.to_csv(
                           os.path.join(OutFolder,"DV_bldg_repair_sample.zip"),
                           index_label=repair_sample_s.columns.name,
                           compression=dict(
                               method='zip',
                               archive_name='DV_bldg_repair_sample.csv'))

                   if 'Statistics' in out_reqs:
                       repair_stats = convert_to_SimpleIndex(
                           describe(repair_sample),
                           axis=1)
                       repair_stats.to_csv(os.path.join(OutFolder,"DV_bldg_repair_stats.csv"),
                                           index_label=repair_stats.columns.name)

                   if np.any(np.isin(
                           ['GroupedSample', 'GroupedStatistics'], out_reqs)):
                       grp_repair = repair_sample.groupby(
                           level=[0, 1, 2], axis=1).sum()

                       if 'GroupedSample' in out_reqs:
                           grp_repair_s = convert_to_SimpleIndex(grp_repair, axis=1)
                           grp_repair_s.to_csv(
                               os.path.join(OutFolder,"DV_bldg_repair_grp.zip"),
                               index_label=grp_repair_s.columns.name,
                               compression=dict(
                                   method='zip',
                                   archive_name='DV_bldg_repair_grp.csv'))

                       if 'GroupedStatistics' in out_reqs:
                           grp_stats_DV = convert_to_SimpleIndex(
                               describe(grp_repair), axis=1)
                           grp_stats_DV.to_csv(os.path.join(OutFolder,"DV_bldg_repair_grp_stats.csv"),
                                            index_label=grp_stats_DV.columns.name)

                   if np.any(np.isin(['AggregateSample',
                                      'AggregateStatistics'], out_reqs)):                       

                       if 'AggregateSample' in out_reqs:
                           agg_repair_s = convert_to_SimpleIndex(self.agg_repair, axis=1)
                           agg_repair_s.to_csv(
                               os.path.join(OutFolder,"DV_bldg_repair_agg.zip"),
                               index_label=agg_repair_s.columns.name,
                               compression=dict(
                                   method='zip',
                                   archive_name='DV_bldg_repair_agg.csv'))

                       if 'AggregateStatistics' in out_reqs:
                           agg_stats = convert_to_SimpleIndex(
                               describe(self.agg_repair), axis=1)
                           agg_stats.to_csv(os.path.join(OutFolder,"DV_bldg_repair_agg_stats.csv"),
                                            index_label=agg_stats.columns.name)
                           
        self.Statistics_demand = describe(demand_sample)
        self.Statistics_dmg_grp = grp_stats_damage
        self.Statistics_DV_grp = grp_stats_DV
        self.Statistics_loss = summary_stats
        self.Statistics_repr = repair_stats
        
        
        
    def getLossByCmp(self, plot):
        
        cmpLoss = self.repair_sample.groupby(level=[0, 2], axis=1).sum()['COST'].iloc[:, :-2]


        if plot == 'plot':
            
            # we add 100 to the loss values to avoid having issues with zeros when creating a log plot
            cmpLoss += 100
            
            fig = px.box(y=np.tile(cmpLoss.columns, cmpLoss.shape[0]), 
                   x=cmpLoss.values.flatten(), 
                   color = [c[0] for c in cmpLoss.columns]*cmpLoss.shape[0],
                   orientation = 'h',
                   labels={
                       'x':'Aggregate repair cost [2011 USD]',
                       'y':'Component ID',
                       'color': 'Component Group'
               },
               title=f'Range of repair cost realizations by component type',
               log_x=True,
               height=1500)
            
            fig.show()
            
        return cmpLoss
        
    def getLossBySt(self, plot):
                 
        stLoss = self.repair_sample['COST'].groupby('loc', axis=1).sum().describe([0.1,0.5,0.9]).iloc[:, 1:]
        
        if plot == 'plot':
            fig = px.pie(values=stLoss.loc['mean'],
                   names=[f'floor {c}' if int(c)<5 else 'roof' for c in stLoss.columns],
                   title='Contribution of each floor to the average non-collapse repair costs',
                   height=500,
                   hole=0.4
                  )
            
            fig.update_traces(textinfo='percent+label')
            
            fig.show()
            
        return stLoss
    
    def getLossBySubsystem(self, plot):

        index = self.PAL._damage.damage_params.iloc[:,2]
        index = pd.Series.to_list(index)
        
        # delete SA from components EDP
        del index[-1]
        
        loss = self.repair_sample['COST'].groupby('loss',level=[0],axis=1).sum()
        loss.columns = pd.MultiIndex.from_arrays([np.array(index).T.tolist(), loss.columns])
        loss.columns = loss.columns.rename('EDP', level=0)
 
        dissLoss = loss.groupby('EDP',level=0,axis=1).sum().describe([0.1,0.5,0.9])
        
        if plot == 'plot':
            fig = px.pie(values=dissLoss.loc['mean'],
                   names=dissLoss.columns,
                   title='Contribution of each subsystem to mean non-collapse repair cost',
                   height=500,
                   hole=0.4
                  )
            
            fig.update_traces(textinfo='percent+label')
            
            fig.show()
        
        return dissLoss
       
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


num_stripes = 1
lossCmp=[]
lossSubsyst=[]
lossSt=[]

for stripe_num in range(num_stripes):

    Ass = LossAssesment(str(stripe_num + 1),r'C:\Users\dsuas\My_Drive\UME\3.Codes\LossAssessmentPelicun\input.json')
    Ass = Ass.get_input()
    Ass = Ass.run_assesment()
    Ass = Ass.processResults()
    
    
    lossCmp[stripe_num] = Ass.getLossByCmp('plot')
    lossSubsyst[stripe_num] = Ass.getLossBySubsystem('plot')
    lossSt[stripe_num] = Ass.getLossBySt('plot')


