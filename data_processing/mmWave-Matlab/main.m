%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
%
%
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions
%   are met:
%
%     Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the
%     documentation and/or other materials provided with the
%     distribution.
%
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%


clearvars
close all

%%  filepath setting
s_data_beg = ['Your Data Path', '/RT-pose/sequences/'];
s_data_end =  '/radar/bin/';
s_output_beg = s_data_beg;
s_output_end = '/radar/';

sequences = 0:241;

for i = sequences

    % The following sequences were unexpectedly damaged
    if i == 44 or 68 or 107 or 155
        continue
    
    s = string(i);
    mat_folder = fullfile(s_data_beg, s , '/radar/mat/');
    if ~exist(mat_folder, 'dir')
        mkdir(mat_folder)
    end
        
    data_folder = fullfile(s_data_beg,s,s_data_end);
    output_folder = mat_folder;
    param_name = 'main/cascade/input/hardware_param.m';
    calib_name = 'main/cascade/input/calibrateResults_high.mat';
       
    disp(data_folder);
    process_cas(data_folder, output_folder, param_name, calib_name);
        
end