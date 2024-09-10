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
%

function process_cas(data_folder, output_folder, param_path, calib_path)

    addpath(genpath('.'))
    LOG_ON = 1; % 1: log10 scale; 0: linear scale
    
    %load calibration parameters
    load(calib_path)
    
    % simTopObj is used for top level parameter parsing and data loading and saving
    simTopObj           = simTopCascade('pfile', param_path);
    calibrationObj      = calibrationCascade('pfile', param_path, 'calibrationfilePath', calib_path);
    rangeFFTObj         = rangeProcCascade('pfile', param_path);
    DopplerFFTObj       = DopplerProcClutterRemove('pfile', param_path);
    detectionObj        = CFAR_CASO('pfile', param_path);
    DOAObj              = DOACascade('pfile', param_path);
    
    % get system level variables
    platform            = simTopObj.platform;
    numValidFrames      = simTopObj.totNumFrames;
    cnt = 1;
    frameCountGlobal = 0;

    
    % Get Unique File Idxs in the "data_folder"   
    [fileIdx_unique] = getUniqueFileIdx(data_folder);
    for i_file = 1:(length(fileIdx_unique))
        
        % Get File Names for the Master, Slave1, Slave2, Slave3
        [fileNameStruct]= getBinFileNames_withIdx(data_folder, fileIdx_unique{i_file});
        
        %pass the Data File to the calibration Object
        calibrationObj.binfilePath = fileNameStruct;
        
        % Get Valid Number of Frames 
        [numValidFrames dataFileSize] = getValidNumFrames(fullfile(data_folder, fileNameStruct.masterIdxFile));
        %intentionally skip the first frame due to TDA2

        for frameIdx = 2:1:numValidFrames  %numFrames_toRun
            filename = ['4dTensor-Frame' num2str(frameIdx) '.mat'];
            filepath = fullfile(output_folder, filename);

            % read and calibrate raw ADC data
            % ADC: (n_sample, n_chirp, n_Rx, n_Tx)
            calibrationObj.frameIdx = frameIdx;
            frameCountGlobal = frameCountGlobal + 1;
            adcData = datapath(calibrationObj);
            
            % RX Channel re-ordering
            adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:);
            
            %only take TX and RXs required for MIMO data analysis
            
            if mod(frameIdx, 10)==1
                fprintf('Processing %3d frame...\n', frameIdx);
            end
            
            %perform 2D FFT
            rangeFFTOut = [];
            DopplerFFTOut = [];

            for i_tx = 1: size(adcData,4)
                % range FFT
                rangeFFTOut(:,:,:,i_tx)     = datapath(rangeFFTObj, adcData(:,:,:,i_tx));
                
                % Doppler FFT
                DopplerFFTOut(:,:,:,i_tx)   = datapath(DopplerFFTObj, rangeFFTOut(:,:,:,i_tx));
                
            end

            DopplerFFTOut = reshape(DopplerFFTOut,size(DopplerFFTOut,1), size(DopplerFFTOut,2), size(DopplerFFTOut,3)*size(DopplerFFTOut,4));

            STATIC_ONLY = 1;
            minRangeBinKeep = 1;
            rightRangeBinDiscard = 0;

            Process_4DRT(detectionObj.rangeBinSize, DopplerFFTOut,...
                          length(calibrationObj.IdTxForMIMOProcess),...
                          length(calibrationObj.RxForMIMOProcess), ...
                          LOG_ON, STATIC_ONLY, minRangeBinKeep,...
                          rightRangeBinDiscard, DOAObj.D, output_folder, frameIdx);

            cnt = cnt + 1;
%             toc
        end
    end

end