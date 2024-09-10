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

function Process_4DRT(range_resolution, radar_data_pre_3dfft,TDM_MIMO_numTX,numRxAnt,...
    LOG, STATIC_ONLY, minRangeBinKeep,  rightRangeBinDiscard, D, output_folder, frameIdx)

dopplerFFTSize = size(radar_data_pre_3dfft,2);
rangeFFTSize = size(radar_data_pre_3dfft,1);
angleFFTSize = 128;
elevationFFTSize = 32;

%4-D Tensor take out same data form viutral antenna table of D (DoAobj)
D = D + 1;
apertureLen_azim = max(D(:,1));
apertureLen_elev = max(D(:,2));
sig_4D = zeros(size(radar_data_pre_3dfft,1), size(radar_data_pre_3dfft,2), apertureLen_azim, apertureLen_elev);
for i=1:size(radar_data_pre_3dfft,1)
    for j = 1:size(radar_data_pre_3dfft,2)
        sig = radar_data_pre_3dfft(i,j,:);
        sig_2D = zeros(apertureLen_azim,apertureLen_elev);
        for i_line = 1:apertureLen_elev
            ind = find(D(:,2) == i_line);
            D_sel = D(ind,1);
            sig_sel = sig(ind);
            [val indU] = unique(D_sel);
    
            sig_2D(D_sel(indU),i_line) = sig_sel(indU);
        end
        sig_4D(i,j,:,:) = sig_2D;
    end    
end

sig_4D = permute(sig_4D, [2 1 3 4]);

azimfft = fft(sig_4D,angleFFTSize, 3);
azimfft = fftshift(azimfft,3);
matr4 = fft(azimfft, elevationFFTSize, 4);
matr4 = fftshift(matr4,4);
matr4 = cast(matr4,'single');

filename = ['4dTensor-Frame' num2str(frameIdx)];
path = fullfile(output_folder, filename);

save(path, "matr4", '-v7.3');

end