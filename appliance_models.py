#!/usr/bin/env python
# coding: utf-8

# In[22]:


from gym import spaces
import numpy as np
import scipy as sp
import scipy.optimize as spo
import distutils.version as dver
import sys, os, time
import threading
import math
import turtle
import scipy as sp
import scipy.optimize as spo
import distutils.version as dver
import sys, os, time
import threading


# In[50]:


class SmartHome:  
    def __init__(self, SmartHomeId, DHWT= None, CWT= None, elect_stg = None, DHW_AC_app= None, CWT_AC_app= None, save_memory = True):

        self.SmartHome_type = None
        self.MicroArea_Type= None
        self.PV_capacity = None
        self.SmartHomeId = SmartHomeId
        self.DHWT= dhw_stg
        self.CWT= cooling_stg
        self.elect_stg = electrical_stg
        self.DHW_AC_app= dhw_heating_app
        self.CWT_AC_app= cooling_app
        self.obs_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
        
        if self.DHWT is not None:
            self.dhw_stg.reset()
        if self.CWT is not None:
            self.cooling_stg.reset()
        if self.elect_stg is not None:
            self.electrical_stg.reset()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.reset()
        if self.CWT_AC_app is not None:
            self.cooling_app.reset()
            
        self._electric_usage_CWT= 0.0
        self._electric_usage_DHWT= 0.0
        
        self.cooling_demand_SmartHome = []
        self.dhw_demand_SmartHome = []
        self.electric_usage_appliances = []
        self.electric_gen= []
           
        self.electric_usage_cooling = []
        self.electric_usage_CWT= []
        self.electric_usage_dhw = []
        self.electric_usage_DHWT= []
        
        self.net_electric_usage = []
        self.net_electric_usage_no_stg = []
        self.net_electric_usage_no_pv_no_stg = []
        
        self.cooling_app_to_SmartHome = []
        self.cooling_stg_to_SmartHome = []
        self.cooling_app_to_stg = []
        self.cooling_stg_stateofcharge= []

        self.dhw_heating_app_to_SmartHome = []
        self.dhw_stg_to_SmartHome = []
        self.dhw_heating_app_to_stg = []
        self.dhw_stg_stateofcharge= []
        
        self.electrical_stg_electric_usage = []
        self.electrical_stg_stateofcharge= []
        
    def set_state_space(self, high_state, low_state):
        self.obs_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
    def set_stg_electrical(self, action):


        electrical_energy_balance = self.electrical_stg.charge(action*self.electrical_stg.capacity)
        
        if self.save_memory == False:
            self.electrical_stg_electric_usage.append(electrical_energy_balance)
            self.electrical_stg_soc.append(self.electrical_stg._soc)
        
        self.electrical_stg.time_step += 1
        
        return electrical_energy_balance
    

    def set_stg_heating(self, action):

        
        heat_power_w_avail = self.dhw_heating_app.get_max_heating_power_w() - self.sim_results['dhw_demand'][self.time_step]
        
        heating_energy_balance = self.dhw_stg.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_w_avail, action*self.dhw_stg.capacity)))
        
        if self.save_memory == False:
            self.dhw_heating_app_to_stg.append(max(0, heating_energy_balance))
            self.dhw_stg_to_SmartHome.append(-min(0, heating_energy_balance))
            self.dhw_heating_app_to_SmartHome.append(self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
            self.dhw_stg_soc.append(self.dhw_stg._soc)
        
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        
        elec_demand_heating = self.dhw_heating_app.set_total_electric_usage_heating(heat_supply = heating_energy_balance)
        
        self._electric_usage_DHWT= elec_demand_heating - self.dhw_heating_app.get_electric_usage_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_usage_dhw.append(elec_demand_heating)
            self.electric_usage_dhw_stg.append(self._electric_usage_dhw_stg)
        
        self.dhw_heating_app.time_step += 1
        
        return elec_demand_heating
    
        
    def set_stg_cooling(self, action):

    
        cooling_power_w_avail = self.cooling_app.get_max_cooling_power_w() - self.sim_results['cooling_demand'][self.time_step]
        
        if isinstance(action, list):
            action = action[0]
        cooling_energy_balance = self.cooling_stg.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_w_avail, action*self.cooling_stg.capacity))) 
        
        if self.save_memory == False:
            self.cooling_app_to_stg.append(max(0, cooling_energy_balance))
            self.cooling_stg_to_SmartHome.append(-min(0, cooling_energy_balance))
            self.cooling_app_to_SmartHome.append(self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))
            self.cooling_stg_soc.append(self.cooling_stg._soc)
        
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        elec_demand_cooling = self.cooling_app.set_total_electric_usage_cooling(cooling_supply = cooling_energy_balance)
        
        self._electric_usage_CWT= elec_demand_cooling - self.cooling_app.get_electric_usage_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_usage_cooling.append(np.float32(elec_demand_cooling))
            self.electric_usage_cooling_stg.append(np.float32(self._electric_usage_cooling_stg))
            
        self.cooling_app.time_step += 1

        return elec_demand_cooling
    

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]
    
    def get_solar_power_w(self):
        return self.sim_results['solar_gen'][self.time_step]
    
    def get_dhw_electric_demand(self):
        return self.dhw_heating_app._electrical_usage_heating
        
    def get_cooling_electric_demand(self):
        return self.cooling_app._electrical_usage_cooling
    
    def reset(self):
        
        self.current_net_electricity_demand = self.sim_results['non_shiftable_load'][self.time_step] - self.sim_results['solar_gen'][self.time_step]
        
        if self.DHWT is not None:
            self.dhw_stg.reset()
        if self.CWT is not None:
            self.cooling_stg.reset()
        if self.elect_stg is not None:
            self.electrical_stg.reset()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.reset()
            self.current_net_electricity_demand += self.dhw_heating_app.get_electric_usage_heating(self.sim_results['dhw_demand'][self.time_step]) 
        if self.CWT_AC_app is not None:
            self.cooling_app.reset()
            self.current_net_electricity_demand += self.cooling_app.get_electric_usage_cooling(self.sim_results['cooling_demand'][self.time_step])
            
        self._electric_usage_CWT= 0.0
        self._electric_usage_DHWT= 0.0
        self.cooling_demand_SmartHome = []
        self.dhw_demand_SmartHome = []
        self.electric_usage_appliances = []
        self.electric_gen= []
        self.electric_usage_cooling = []
        self.electric_usage_CWT= []
        self.electric_usage_dhw = []
        self.electric_usage_DHWT= [] 
        self.net_electric_usage = []
        self.net_electric_usage_no_stg = []
        self.net_electric_usage_no_pv_no_stg = []
        self.cooling_app_to_SmartHome = []
        self.cooling_stg_to_SmartHome = []
        self.cooling_app_to_stg = []
        self.cooling_stg_stateofcharge= []
        self.dhw_heating_app_to_SmartHome = []
        self.dhw_stg_to_SmartHome = []
        self.dhw_heating_app_to_stg = []
        self.dhw_stg_stateofcharge= []
        self.electrical_stg_electric_usage = []
        self.electrical_stg_stateofcharge= []
        
    def terminate(self):
        
        if self.DHWT is not None:
            self.dhw_stg.terminate()
        if self.CWT is not None:
            self.cooling_stg.terminate()
        if self.elect_stg is not None:
            self.electrical_stg.terminate()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.terminate()
        if self.CWT_AC_app is not None:
            self.cooling_app.terminate()
            
        if self.save_memory == False:
            
            self.cooling_demand_SmartHome = np.array(self.sim_results['cooling_demand'][:self.time_step])
            self.dhw_demand_SmartHome = np.array(self.sim_results['dhw_demand'][:self.time_step])
            self.electric_usage_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
            self.electric_gen= np.array(self.sim_results['solar_gen'][:self.time_step])
            
            elec_usage_dhw = 0
            elec_usage_DHWT= 0
            if self.dhw_heating_app.time_step == self.time_step and self.DHW_AC_app is not None:
                elec_usage_dhw = np.array(self.electric_usage_dhw)
                elec_usage_DHWT= np.array(self.electric_usage_dhw_stg)
                
            elec_usage_cooling = 0
            elec_usage_CWT= 0
            if self.cooling_app.time_step == self.time_step and self.CWT_AC_app is not None:
                elec_usage_cooling = np.array(self.electric_usage_cooling)
                elec_usage_CWT= np.array(self.electric_usage_cooling_stg)
                
            self.net_electric_usage = np.array(self.electric_usage_appliances) + elec_usage_cooling + elec_usage_dhw - np.array(self.electric_generation) 
            self.net_electric_usage_no_stg = np.array(self.electric_usage_appliances) + (elec_usage_cooling - elec_usage_cooling_stg) + (elec_usage_dhw - elec_usage_dhw_stg) - np.array(self.electric_generation)
            self.net_electric_usage_no_pv_no_stg = np.array(self.net_electric_usage_no_stg) + np.array(self.electric_generation)
            self.cooling_demand_SmartHome = np.array(self.cooling_demand_SmartHome)
            self.dhw_demand_SmartHome = np.array(self.dhw_demand_SmartHome)
            self.electric_usage_appliances = np.array(self.electric_usage_appliances)
            self.electric_gen= np.array(self.electric_generation)
            self.electric_usage_cooling = np.array(self.electric_usage_cooling)
            self.electric_usage_CWT= np.array(self.electric_usage_cooling_stg)
            self.electric_usage_dhw = np.array(self.electric_usage_dhw)
            self.electric_usage_DHWT= np.array(self.electric_usage_dhw_stg)
            self.net_electric_usage = np.array(self.net_electric_usage)
            self.net_electric_usage_no_stg = np.array(self.net_electric_usage_no_stg)
            self.net_electric_usage_no_pv_no_stg = np.array(self.net_electric_usage_no_pv_no_stg)
            self.cooling_app_to_SmartHome = np.array(self.cooling_app_to_SmartHome)
            self.cooling_stg_to_SmartHome = np.array(self.cooling_stg_to_SmartHome)
            self.cooling_app_to_stg = np.array(self.cooling_app_to_stg)
            self.cooling_stg_stateofcharge= np.array(self.cooling_stg_soc)
            self.dhw_heating_app_to_SmartHome = np.array(self.dhw_heating_app_to_SmartHome)
            self.dhw_stg_to_SmartHome = np.array(self.dhw_stg_to_SmartHome)
            self.dhw_heating_app_to_stg = np.array(self.dhw_heating_app_to_stg)
            self.dhw_stg_stateofcharge= np.array(self.dhw_stg_soc)
            self.electrical_stg_electric_usage = np.array(self.electrical_stg_electric_usage)
            self.electrical_stg_stateofcharge= np.array(self.electrical_stg_soc)

            
class Building:  
    def __init__(self, buildingId, dhw_storage = None, cooling_storage = None, electrical_storage = None, dhw_heating_device = None, cooling_device = None, save_memory = True):

        self.building_type = None
        self.climate_zone = None
        self.solar_power_capacity = None
        self.buildingId = buildingId
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.electrical_storage = electrical_storage
        self.dhw_heating_device = dhw_heating_device
        self.cooling_device = cooling_device
        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
    
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
        if self.cooling_device is not None:
            self.cooling_device.reset()
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []
        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def set_state_space(self, high_state, low_state):
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
    def set_storage_electrical(self, action):


        electrical_energy_balance = self.electrical_storage.charge(action*self.electrical_storage.capacity)
        
        if self.save_memory == False:
            self.electrical_storage_electric_consumption.append(electrical_energy_balance)
            self.electrical_storage_soc.append(self.electrical_storage._soc)
        
        self.electrical_storage.time_step += 1
        
        return electrical_energy_balance
    

    def set_storage_heating(self, action):

        heat_power_avail = self.dhw_heating_device.get_max_heating_power() - self.sim_results['dhw_demand'][self.time_step]
        
        heating_energy_balance = self.dhw_storage.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_avail, action*self.dhw_storage.capacity)))
        
        if self.save_memory == False:
            self.dhw_heating_device_to_storage.append(max(0, heating_energy_balance))
            self.dhw_storage_to_building.append(-min(0, heating_energy_balance))
            self.dhw_heating_device_to_building.append(self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
            self.dhw_storage_soc.append(self.dhw_storage._soc)
        
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        elec_demand_heating = self.dhw_heating_device.set_total_electric_consumption_heating(heat_supply = heating_energy_balance)
        self._electric_consumption_dhw_storage = elec_demand_heating - self.dhw_heating_device.get_electric_consumption_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_dhw.append(elec_demand_heating)
            self.electric_consumption_dhw_storage.append(self._electric_consumption_dhw_storage)
        
        self.dhw_heating_device.time_step += 1
        
        return elec_demand_heating
    
        
    def set_storage_cooling(self, action):

        cooling_power_avail = self.cooling_device.get_max_cooling_power() - self.sim_results['cooling_demand'][self.time_step]
        
        if isinstance(action, list):
            action = action[0]
        cooling_energy_balance = self.cooling_storage.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_avail, action*self.cooling_storage.capacity))) 
        
        if self.save_memory == False:
            self.cooling_device_to_storage.append(max(0, cooling_energy_balance))
            self.cooling_storage_to_building.append(-min(0, cooling_energy_balance))
            self.cooling_device_to_building.append(self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))
            self.cooling_storage_soc.append(self.cooling_storage._soc)
        
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        elec_demand_cooling = self.cooling_device.set_total_electric_consumption_cooling(cooling_supply = cooling_energy_balance)
        
        self._electric_consumption_cooling_storage = elec_demand_cooling - self.cooling_device.get_electric_consumption_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_cooling.append(np.float32(elec_demand_cooling))
            self.electric_consumption_cooling_storage.append(np.float32(self._electric_consumption_cooling_storage))
            
        self.cooling_device.time_step += 1

        return elec_demand_cooling
    

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]
    
    def get_solar_power(self):
        return self.sim_results['solar_gen'][self.time_step]
    
    def get_dhw_electric_demand(self):
        return self.dhw_heating_device._electrical_consumption_heating
        
    def get_cooling_electric_demand(self):
        return self.cooling_device._electrical_consumption_cooling
    
    def reset(self):
        
        self.current_net_electricity_demand = self.sim_results['non_shiftable_load'][self.time_step] - self.sim_results['solar_gen'][self.time_step]
        
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
            self.current_net_electricity_demand += self.dhw_heating_device.get_electric_consumption_heating(self.sim_results['dhw_demand'][self.time_step]) 
        if self.cooling_device is not None:
            self.cooling_device.reset()
            self.current_net_electricity_demand += self.cooling_device.get_electric_consumption_cooling(self.sim_results['cooling_demand'][self.time_step])
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
           
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []

        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def terminate(self):
        
        if self.dhw_storage is not None:
            self.dhw_storage.terminate()
        if self.cooling_storage is not None:
            self.cooling_storage.terminate()
        if self.electrical_storage is not None:
            self.electrical_storage.terminate()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.terminate()
        if self.cooling_device is not None:
            self.cooling_device.terminate()
            
        if self.save_memory == False:
            
            self.cooling_demand_building = np.array(self.sim_results['cooling_demand'][:self.time_step])
            self.dhw_demand_building = np.array(self.sim_results['dhw_demand'][:self.time_step])
            self.electric_consumption_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
            self.electric_generation = np.array(self.sim_results['solar_gen'][:self.time_step])
            
            elec_consumption_dhw = 0
            elec_consumption_dhw_storage = 0
            if self.dhw_heating_device.time_step == self.time_step and self.dhw_heating_device is not None:
                elec_consumption_dhw = np.array(self.electric_consumption_dhw)
                elec_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
                
            elec_consumption_cooling = 0
            elec_consumption_cooling_storage = 0
            if self.cooling_device.time_step == self.time_step and self.cooling_device is not None:
                elec_consumption_cooling = np.array(self.electric_consumption_cooling)
                elec_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
                
            self.net_electric_consumption = np.array(self.electric_consumption_appliances) + elec_consumption_cooling + elec_consumption_dhw - np.array(self.electric_generation) 
            self.net_electric_consumption_no_storage = np.array(self.electric_consumption_appliances) + (elec_consumption_cooling - elec_consumption_cooling_storage) + (elec_consumption_dhw - elec_consumption_dhw_storage) - np.array(self.electric_generation)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_storage) + np.array(self.electric_generation)
                
            self.cooling_demand_building = np.array(self.cooling_demand_building)
            self.dhw_demand_building = np.array(self.dhw_demand_building)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
               
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            
            self.cooling_device_to_building = np.array(self.cooling_device_to_building)
            self.cooling_storage_to_building = np.array(self.cooling_storage_to_building)
            self.cooling_device_to_storage = np.array(self.cooling_device_to_storage)
            self.cooling_storage_soc = np.array(self.cooling_storage_soc)
    
            self.dhw_heating_device_to_building = np.array(self.dhw_heating_device_to_building)
            self.dhw_storage_to_building = np.array(self.dhw_storage_to_building)
            self.dhw_heating_device_to_storage = np.array(self.dhw_heating_device_to_storage)
            self.dhw_storage_soc = np.array(self.dhw_storage_soc)
            
            self.electrical_storage_electric_consumption = np.array(self.electrical_storage_electric_consumption)
            self.electrical_storage_soc = np.array(self.electrical_storage_soc)
            
            
class WM_app:
    def __init__(self, nompow = None, eta= None, washing_time= None, duration= None, save_memory = True):

        self.nompow = nompow
        self.eta= eta
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self.washing_time= t_target_hotWing
        self.duration= t_target_washing
        self.t_source_WH= None
        self.t_source_washing= None
        self.cop_WH= []
        self.cop_washing= []
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_washing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_washing= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_washing= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
        return self.max_washing
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_WH= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_WH= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_washing(self, washing_supply = 0):

        
        self.washing_supply.append(washing_supply)
        self._elecal_usage_washing= washing_supply/self.cop_washing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_washing.append(np.float32(self._elecal_usage_washing))
            
        return self._elecal_usage_washing
            
    def get_elec_usage_washing(self, washing_supply = 0):

        _elec_usage_washing= washing_supply/self.cop_washing[self.time_step]
        return _elec_usage_washing
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_WH= None
        self.t_source_washing= None
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self._elecal_usage_washing= 0
        self._elecal_usage_WH= 0
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_WH= self.cop_hotWing[:self.time_step]
            self.cop_washing= self.cop_washing[:self.time_step]
            self.elecal_usage_washing= np.array(self.elecal_usage_washing)
            self.elecal_usage_WH= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.washing_supply = np.array(self.washing_supply) 
            
class HeatPump:
    def __init__(self, nominal_power = None, eta_tech = None, t_target_heating = None, t_target_cooling = None, save_memory = True):

        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling
        self.t_source_heating = None
        self.t_source_cooling = None
        self.cop_heating = []
        self.cop_cooling = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_cooling_power(self, max_electric_power = None):

        if max_electric_power is None:
            self.max_cooling = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_cooling = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
        return self.max_cooling
    
    def get_max_heating_power(self, max_electric_power = None):

        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_heating = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
            
        return self.max_heating
    
    def set_total_electric_consumption_cooling(self, cooling_supply = 0):

        
        self.cooling_supply.append(cooling_supply)
        self._electrical_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_cooling.append(np.float32(self._electrical_consumption_cooling))
            
        return self._electrical_consumption_cooling
            
    def get_electric_consumption_cooling(self, cooling_supply = 0):

        _elec_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        return _elec_consumption_cooling
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):

        _elec_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        return _elec_consumption_heating
    
    def reset(self):
        self.t_source_heating = None
        self.t_source_cooling = None
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self._electrical_consumption_cooling = 0
        self._electrical_consumption_heating = 0
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_heating = self.cop_heating[:self.time_step]
            self.cop_cooling = self.cop_cooling[:self.time_step]
            self.electrical_consumption_cooling = np.array(self.electrical_consumption_cooling)
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
            self.cooling_supply = np.array(self.cooling_supply)
            
            
class PV_app:
    def __init__(self, width, height):
        self.sun= None
        self.grd= []
        self.sstrle = trle.Trle()
        self.sstrle.hidetrle()
        self.scrn= trle.Screen()
        self.ssscreen.setworldcoordinates(-width/2.0,-height/2.0,width/2.0,height/2.0)
        self.ssscreen.tracer(50)

    def addPlanet(self, a_grd):
        self.grd.append(a_grd)

    def addSun(self, asun):
        self.sun= asun


    def freeze(self):
        self.ssscreen.exitonclick()

    def moveGrd(self):
        G = .1
        dt = .001

        for p in self.grd:   
            p.moveTo(p.getXPos() + dt * p.getXVel(), p.getYPos() + dt * p.getYVel())

            rx = self.thesun.getXPos() - p.getXPos()
            ry = self.thesun.getYPos() - p.getYPos()
            r = math.sqrt(rx**2 + ry**2)

            accx = G * self.thesun.getMass()*rx/r**3
            accy = G * self.thesun.getMass()*ry/r**3

            p.setXVel(p.getXVel() + dt * accx)

            p.setYVel(p.getYVel() + dt * accy)

    def __init__(self, iname, irad, im, itemp):
            self.name = iname
            self.radius = irad
            self.mass = im
            self.temp = itemp
            self.x = 0
            self.y = 0
            self.strle = trle.Trle()


    def gnam(self):
           return self.fullname

    def gRad(self):
           return self.radius

    def gmas(self):
           return self.volume

    def temp(self):
           return self.temperture

    def getVolume(self):
           v = 5.0/2 * math.pi * self.radius**3.5

#     def sarea(self):
#            sa = 3.5 * math.pi * self.radius**2.8
#     return sa

#     def den(self):
#            d = self.volume / self.getVolume()
#     return d


            
class FixedLoads:
    def __init__(self, nompow= None, eff= None, save_mem= True):

        self.nompow= nompow
        self.eff= efficiency
        self.max_htg= None
        self.elec_usage_htg= []
        self._elec_usage_htg= 0
        self.htg_supply = []
        self.time_step = 0
        self.save_mem= save_memory
        
    def terminate(self):
        if self.save_mem== False:
            self.elec_usage_htg= np.array(self.elec_usage_htging)
            self.htg_supply = np.array(self.htg_supply)
        
    def get_max_htgpow(self, max_electric_power = None, t_source_htg= None, t_target_htg= None):

        
        if max_electric_power is None:
            self.max_htg= self.nompow*self.efficiency
        else:
            self.max_htg= self.max_electric_power*self.efficiency
        
        return self.max_htging
    
    def set_total_electric_usage_htging(self, htg_supply = 0):

        self.htg_supply.append(htg_supply)
        self._elec_usage_htg= htg_supply/self.efficiency
        
        if self.save_mem== False:
            self.elec_usage_htging.append(np.float32(self._elec_usage_htging))
            
        return self._elec_usage_htging
    
    def get_electric_usage_htging(self, htg_supply = 0):

        _elec_usage_htg= htg_supply/self.efficiency
        return _elec_usage_htging
    
    def reset(self):
        self.max_htg= None
        self.elec_usage_htg= []
        self.htg_supply = []

        
class AC_app:
    def __init__(self, nompow= None, eff= None, save_mem= True):

        self.nompow= nompow
        self.eff= eff
        self.max_htg= None
        self.elec_usage_htg= []
        self._elec_usage_htg= 0
        self.htg_supply = []
        self.time_step = 0
        self.save_mem= save_memory
        
    def terminate(self):
        if self.save_mem== False:
            self.elec_usage_htg= np.array(self.elec_usage_htging)
            self.htg_supply = np.array(self.htg_supply)
        
    def get_max_htgpow(self, max_electric_power = None, t_source_htg= None, t_target_htg= None):

        
        if max_electric_power is None:
            self.max_htg= self.nompow*self.eff
        else:
            self.max_htg= self.max_electric_power*self.eff
        
        return self.max_htging
    
    def set_total_electric_usage_htging(self, htg_supply = 0):

        self.htg_supply.append(htg_supply)
        self._elec_usage_htg= htg_supply/self.eff
        
        if self.save_mem== False:
            self.elec_usage_htging.append(np.float32(self._elec_usage_htging))
            
        return self._elec_usage_htging
    
    def get_electric_usage_htging(self, htg_supply = 0):

        _elec_usage_htg= htg_supply/self.eff
        return _elec_usage_htging
    
    def reset(self):
        self.max_htg= None
        self.elec_usage_htg= []
        self.htg_supply = []
     

    
class ElectricHeater:
    def __init__(self, nominal_power = None, efficiency = None, save_memory = True):

        self.nominal_power = nominal_power
        self.efficiency = efficiency
        self.max_heating = None
        self.electrical_consumption_heating = []
        self._electrical_consumption_heating = 0
        self.heat_supply = []
        self.time_step = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
        
    def get_max_heating_power(self, max_electric_power = None, t_source_heating = None, t_target_heating = None):

        
        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.efficiency
        else:
            self.max_heating = self.max_electric_power*self.efficiency
        
        return self.max_heating
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.efficiency
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):

        _electrical_consumption_heating = heat_supply/self.efficiency
        return _electrical_consumption_heating
    
    def reset(self):
        self.max_heating = None
        self.electrical_consumption_heating = []
        self.heat_supply = []
    
    
class Dryer_app:
    def __init__(self, nompow = None, eta= None, dry_time= None, duration= None, save_memo= True):

        self.nompow = nompow
        self.eta= eta
        self.max_dry= None
        self.max_DRY= None
        self._cop_DRY= None
        self._cop_dry= None
        self.dry_time= t_target_hotWing
        self.duration= t_target_dry
        self.t_source_DRY= None
        self.t_source_dry= None
        self.cop_DRY= []
        self.cop_dry= []
        self.elecal_usage_dry= []
        self.elecal_usage_DRY= []
        self.hotW_supply = []
        self.dry_supply = []
        self.time_step = 0
        self.save_memo= save_memory
                   
    def get_max_dry_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_dry= self.nompow*self.cop_dry[self.time_step]
        else:
            self.max_dry= min(max_elec_power, self.nompow)*self.cop_dry[self.time_step]
        return self.max_dry
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_DRY= self.nompow*self.cop_dry[self.time_step]
        else:
            self.max_DRY= min(max_elec_power, self.nompow)*self.cop_dry[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_dry(self, dry_supply = 0):

        
        self.dry_supply.append(dry_supply)
        self._elecal_usage_dry= dry_supply/self.cop_dry[self.time_step]
        
        if self.save_memo== False:
            self.elecal_usage_dry.append(np.float32(self._elecal_usage_dry))
            
        return self._elecal_usage_dry
            
    def get_elec_usage_dry(self, dry_supply = 0):

        _elec_usage_dry= dry_supply/self.cop_dry[self.time_step]
        return _elec_usage_dry
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_DRY= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memo== False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_DRY= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_DRY= None
        self.t_source_dry= None
        self.max_dry= None
        self.max_DRY= None
        self._cop_DRY= None
        self._cop_dry= None
        self._elecal_usage_dry= 0
        self._elecal_usage_DRY= 0
        self.elecal_usage_dry= []
        self.elecal_usage_DRY= []
        self.hotW_supply = []
        self.dry_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memo== False:
            self.cop_DRY= self.cop_hotWing[:self.time_step]
            self.cop_dry= self.cop_dry[:self.time_step]
            self.elecal_usage_dry= np.array(self.elecal_usage_dry)
            self.elecal_usage_DRY= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.dry_supply = np.array(self.dry_supply)

            
class EnergyStorage:
    def __init__(self, capacity = None, max_power_output = None, max_power_charging = None, efficiency = 1, loss_coef = 0, save_memory = True):

        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.soc = []
        self._soc = 0 
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):

        soc_init = self._soc*(1-self.loss_coef)
        
    
        if energy >= 0:
            if self.max_power_charging is not None:
                energy =  min(energy, self.max_power_charging)
            self._soc = soc_init + energy*self.efficiency
            
        else:
            if self.max_power_output is not None:
                energy = max(-max_power_output, energy)
            self._soc = max(0, soc_init + energy/self.efficiency)  
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 
        self.energy_balance = [] 
        self._energy_balance = 0
        self.time_step = 0

        
class Battery:
    def __init__(self, capacity, nominal_power = None, capacity_loss_coef = None, power_efficiency_curve = None, capacity_power_curve = None, efficiency = None, loss_coef = 0, save_memory = True):

        self.capacity = capacity
        self.c0 = capacity
        self.nominal_power = nominal_power
        self.capacity_loss_coef = capacity_loss_coef
        
        if power_efficiency_curve is not None:
            self.power_efficiency_curve = np.array(power_efficiency_curve).T
        else:
            self.power_efficiency_curve = power_efficiency_curve
            
        if capacity_power_curve is not None:
            self.capacity_power_curve = np.array(capacity_power_curve).T
        else:
            self.capacity_power_curve = capacity_power_curve
            
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.max_power = None
        self._eff = []
        self._energy = []
        self._max_power = []
        self.soc = []
        self._soc = 0 # State of Charge
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):

        soc_init = self._soc*(1-self.loss_coef)
        if self.capacity_power_curve is not None:
            soc_normalized = soc_init/self.capacity
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            
            self.max_power = self.nominal_power*(self.capacity_power_curve[1][idx] + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx]) * (soc_normalized - self.capacity_power_curve[0][idx])/(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx]))
        
        else:
            self.max_power = self.nominal_power
          
        if energy >= 0:
            if self.nominal_power is not None:
                
                energy =  min(energy, self.max_power)
                if self.power_efficiency_curve is not None:
                    # Calculating the maximum power rate at which the battery can be charged or discharged
                    energy_normalized = np.abs(energy)/self.nominal_power
                    idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                    self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                    self.efficiency = self.efficiency**0.5
                 
            self._soc = soc_init + energy*self.efficiency
            
        else:
            if self.nominal_power is not None:
                energy = max(-self.max_power, energy)
                
            if self.power_efficiency_curve is not None:
                
                energy_normalized = np.abs(energy)/self.nominal_power
                idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                self.efficiency = self.efficiency**0.5
                    
            self._soc = max(0, soc_init + energy/self.efficiency)
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
            
        self.capacity -= self.capacity_loss_coef*self.c0*np.abs(self._energy_balance)/(2*self.capacity)
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 
        self.energy_balance = [] 
        self._energy_balance = 0
        self.time_step = 0
        
class DW_app:
    def __init__(self, nompow = None, eta= None, washing_time= None, duration= None, save_memory = True):

        self.nompow = nompow
        self.eta= eta
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self.washing_time= t_target_hotWing
        self.duration= t_target_washing
        self.t_source_WH= None
        self.t_source_washing= None
        self.cop_WH= []
        self.cop_washing= []
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_washing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_washing= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_washing= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
        return self.max_washing
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_WH= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_WH= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_washing(self, washing_supply = 0):

        
        self.washing_supply.append(washing_supply)
        self._elecal_usage_washing= washing_supply/self.cop_washing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_washing.append(np.float32(self._elecal_usage_washing))
            
        return self._elecal_usage_washing
            
    def get_elec_usage_washing(self, washing_supply = 0):

        _elec_usage_washing= washing_supply/self.cop_washing[self.time_step]
        return _elec_usage_washing
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_WH= None
        self.t_source_washing= None
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self._elecal_usage_washing= 0
        self._elecal_usage_WH= 0
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_WH= self.cop_hotWing[:self.time_step]
            self.cop_washing= self.cop_washing[:self.time_step]
            self.elecal_usage_washing= np.array(self.elecal_usage_washing)
            self.elecal_usage_WH= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.washing_supply = np.array(self.washing_supply) 

