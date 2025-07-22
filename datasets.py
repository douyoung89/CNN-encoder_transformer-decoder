# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 density_map_path=None, # Path to the pre-computed density map
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None, # ROI for density map
                 density_size=None):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 
        
        # Load density map if provided
        self.density_map = None
        if density_map_path:
            with open(density_map_path, 'rb') as f:
                self.density_map = pickle.load(f)
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.density_size = density_size
            
            if self.density_map is not None:
                self.density_map_lat_res = (lat_max - lat_min) / self.density_map.shape[0]
                self.density_map_lon_res = (lon_max - lon_min) / self.density_map.shape[1]
                self.max_density_value = self.density_map.max()
                self.min_density_value = self.density_map.min()

    def get_density_index(self, lat, lon):
        """
        Retrieves the discretized density index for a given lat/lon.
        This is an example implementation. You might need to adjust it
        based on how your density map is structured and how density values
        are discretized.
        """
        if self.density_map is None:
            return 0 # Default if no density map is loaded

        # Convert lat/lon to grid indices
        lat_idx = int((lat - self.lat_min) / self.density_map_lat_res)
        lon_idx = int((lon - self.lon_min) / self.density_map_lon_res)

        # Clamp indices to map bounds
        lat_idx = max(0, min(lat_idx, self.density_map.shape[0] - 1))
        lon_idx = max(0, min(lon_idx, self.density_map.shape[1] - 1))

        density_value = self.density_map[lat_idx, lon_idx]
        
        # Discretize density value into self.density_size bins
        if self.max_density_value == self.min_density_value: # Avoid division by zero
            density_index = 0
        else:
            density_index = int(((density_value - self.min_density_value) / 
                                 (self.max_density_value - self.min_density_value)) * self.density_size)
        
        density_index = max(0, min(density_index, self.density_size - 1))

        return density_index
    
    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        
        # Prepare density sequence for encoder input
        density_seq = np.zeros((self.max_seqlen, 1), dtype=np.float32) # Only 1 feature for encoder
        for i in range(seqlen):
            lat, lon = m_v[i,0], m_v[i,1] # Use original lat/lon from trajectory for density lookup
            # Convert normalized lat/lon back to original range for density lookup
            original_lat = lat * (self.lat_max - self.lat_min) + self.lat_min if self.lat_max and self.lat_min else lat
            original_lon = lon * (self.lon_max - self.lon_min) + self.lon_min if self.lon_max and self.lon_min else lon
            density_seq[i,0] = self.get_density_index(original_lat, original_lon)

        encoder_input_density = torch.tensor(density_seq, dtype=torch.float32)
        
        # Decoder input: trajectory features (lat, lon, sog, cog) up to seqlen-1
        decoder_input_seq = torch.tensor(m_v, dtype=torch.float32)[:seqlen, :]
        decoder_input_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        decoder_input_seq_padded[:seqlen, :] = decoder_input_seq

        # Decoder target: next token prediction. If input is x_0...x_T-1, target is x_1...x_T
        decoder_target_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        if seqlen > 1:
            decoder_target_seq_padded[:seqlen-1, :] = decoder_input_seq[1:seqlen, :]

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen_tensor = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)

        return encoder_input_density, decoder_input_seq_padded, decoder_target_seq_padded, mask, seqlen_tensor, mmsi, time_start

    
class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """
    def __init__(self,
                 l_data,
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 density_map_path=None, # Path to the pre-computed density map
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None, # ROI for density map
                 density_size=None):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """

        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

        # Load density map if provided
        self.density_map = None
        if density_map_path:
            with open(density_map_path, 'rb') as f:
                self.density_map = pickle.load(f)
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.density_size = density_size

            if self.density_map is not None:
                self.density_map_lat_res = (lat_max - lat_min) / self.density_map.shape[0]
                self.density_map_lon_res = (lon_max - lon_min) / self.density_map.shape[1]
                self.max_density_value = self.density_map.max()
                self.min_density_value = self.density_map.min()

    def get_density_index(self, lat, lon):
        """
        Retrieves the discretized density index for a given lat/lon.
        """
        if self.density_map is None:
            return 0 # Default if no density map is loaded

        # Convert lat/lon to grid indices
        lat_idx = int((lat - self.lat_min) / self.density_map_lat_res)
        lon_idx = int((lon - self.lon_min) / self.density_map_lon_res)

        # Clamp indices to map bounds
        lat_idx = max(0, min(lat_idx, self.density_map.shape[0] - 1))
        lon_idx = max(0, min(lon_idx, self.density_map.shape[1] - 1))

        density_value = self.density_map[lat_idx, lon_idx]

        # Discretize density value into self.density_size bins
        if self.max_density_value == self.min_density_value: # Avoid division by zero
            density_index = 0
        else:
            density_index = int(((density_value - self.min_density_value) /
                                 (self.max_density_value - self.min_density_value)) * self.density_size)

        density_index = max(0, min(density_index, self.density_size - 1))

        return density_index

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            encoder_input_density: Tensor of (max_seqlen, 1) containing only density for encoder.
            decoder_input_seq: Tensor of (max_seqlen, 4) [lat, lon, sog, cog] for decoder input.
            decoder_target_seq: Tensor of (max_seqlen, 4) [lat, lon, sog, cog] for decoder target.
            mask: Tensor of (max_seqlen). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)

        # Prepare density sequence for encoder input
        density_seq = np.zeros((self.max_seqlen, 1), dtype=np.float32) # Only 1 feature for encoder
        # Use original lat/lon from trajectory for density lookup
        # For AISDataset_grad, m_v[:,:2] contains lat/lon, not dlat/dlon
        for i in range(seqlen):
            lat, lon = m_v[i,0], m_v[i,1]
            original_lat = lat * (self.lat_max - self.lat_min) + self.lat_min if self.lat_max and self.lat_min else lat
            original_lon = lon * (self.lon_max - self.lon_min) + self.lon_min if self.lon_max and self.lon_min else lon
            density_seq[i,0] = self.get_density_index(original_lat, original_lon)

        encoder_input_density = torch.tensor(density_seq, dtype=torch.float32)

        # Decoder input: trajectory features (lat, lon, sog, cog)
        # The `m_v` here already contains the dlat/dlon in columns 2 and 3 if this is AISDataset_grad.
        # So `decoder_input_seq` will be `[lat, lon, dlat, dlon]`
        decoder_input_seq = torch.tensor(m_v, dtype=torch.float32)[:seqlen, :]
        decoder_input_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        decoder_input_seq_padded[:seqlen, :] = decoder_input_seq

        # Decoder target: next token prediction.
        decoder_target_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        if seqlen > 1:
            decoder_target_seq_padded[:seqlen-1, :] = decoder_input_seq[1:seqlen, :]

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen_tensor = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)

        return encoder_input_density, decoder_input_seq_padded, decoder_target_seq_padded, mask, seqlen_tensor, mmsi, time_start

