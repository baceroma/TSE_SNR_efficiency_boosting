function [SNReff,sig_loss_T1,mtf_xy,mtf_z,FA] = ...
    snr_eff_3dtse(FOV,vox,esp,tsefactor_max,T1,T2)

% Description:
%   Computation of SNR efficiency in 3D TSE for a subspace of varying TSE
%   factor and TR.
%   Assumes low-high profile order
% 
% Input:
% - FOV = 3x1 vector with dimensions of the field of view in all three
%   directions in the order M, P, S
% - vox = 3x1 vector with voxel dimensions in the order M, P, S
% - esp = echo spacing (ms)
% - tsefactor_max = maximum TSE factor allowed for the given geometry
% - T1, T2 = relaxation times of reference tissue (ms)
% 
% Output:
% - SNReff = array containing SNR efficiency for given parameter space
% - mtf = (cell) array containing the calculated modulated transfer
%   functions
% - FA = (cell) array containing the calculated flip angle trains (deg)
% 
% 
% B. Cervantes
% September 2015

% Define arrays which define variable parameter space:
% tsefactor_array = min(tsefactor_bds):1:max(tsefactor_bds);
% shot_ratio_array = min(shot_ratio_bds):0.05:max(shot_ratio_bds);
Nx = FOV(1)/vox(1); % number of samples in x
Ny = FOV(2)/vox(2); %                      y
Nz = FOV(3)/vox(3); %                      z
nr_kyz_pts = Ny*Nz;

% Constant parameters:
maxangles = [120,160]; % assumes fixed center and max angles (here default)
startupechoes = 5; % assumes a fixed number of startup echoes
esp_first = 3*esp; % if ultrashort = ultra
shot_ratio = 0.4; % fraction of shot duration of plateau length 

% Define range of TSE factor:
tsefactor_range = 2:2:tsefactor_max; % min is always 2

% Initialise output variables:
tsefactor = zeros(length(tsefactor_range),1);
TR = zeros(length(tsefactor_range),1);
FA = cell(length(tsefactor_range),1);
mtf_xy = cell(length(tsefactor_range),1);
mtf_z = cell(length(tsefactor_range),1);
sig_loss_T1 = zeros(length(tsefactor_range),1);
SNR = zeros(length(tsefactor_range),1);
scn_time = zeros(length(tsefactor_range),1);
SNReff = zeros(length(tsefactor_range),1);

for ii = 1:length(tsefactor_range)
    tsefactor(ii) = tsefactor_range(ii);
    % Define remaining parameters:
    nr_refoc_pulses = startupechoes + tsefactor(ii);
    esp_vec = [esp_first ones(1,startupechoes+tsefactor(ii))*esp];
    shot_dur = 4.42 + esp_first + esp/2 + esp*(nr_refoc_pulses - 3); % see below
    % RFex_posn = constant = 4.4172 ms
    % nr_echoes = nr_refoc_pulses - 1
    % nr_ME_echos = nr_echoes - 1 = nr_refoc_pulses - 2
    % nr_ME_echo_spacings = nr_ME_echoes - 1 = nr_refoc_pulses - 3
    nr_shots = floor(nr_kyz_pts/nr_refoc_pulses);
    % T1 recovery in interval between shots:
    t_recv = 1.5*T1; % delay after last refocusing pulse and next excitation pulse
    m0_frac = 1 - exp(-t_recv/T1); % fraction of M0 for remaining shots
    % Set TR:
    TR(ii) = shot_dur + t_recv;

    % First shot:
    [s0_shot1,mtf_xy_shot1,~,~,FA{ii},mtf_z_shot1] = ...
    signal_centerkspace(maxangles,esp_vec,shot_dur,...
    shot_ratio,T1,T2,startupechoes,tsefactor(ii));

    % Second shot (and same for remaining):
    mtf_xy_shot2 = epg_forward(FA{ii},length(FA{ii}),esp_vec,T1,T2,m0_frac);
    s0_shot2 = mtf_xy_shot2(startupechoes + 1);
    
    % Output for each value of TSE factor:
    % FA{ii} as calculated above
    mtf_xy{ii} = mtf_xy_shot1; % output without T1 effects
    mtf_z{ii} = mtf_z_shot1;
    sig_loss_T1(ii) = (s0_shot1-s0_shot2)/s0_shot1; % keep track of signal 
                                                    % loss between shots
                                                    % due to T1 effects

    % Calculate SNR efficiency:
    % SNR takes into account signal loss from train of refocusing angles in
    % first shot:
    SNR(ii) = s0_shot1*prod(vox)*sqrt(Nx*Ny*Nz)*sqrt(esp-2); % takes into account
    % voxel size and readout bandwidth, assuming a sampling interval of
    % esp-2ms
    scn_time(ii) = TR(ii)*nr_shots;
    SNReff(ii) = SNR(ii)/sqrt(TR(ii)*nr_shots); % SNR/sqrt(scan time)
end

end

function [s0,mtf_xy,psf,fwhm,FA,mtf_z] = ...
    signal_centerkspace(maxangles,esp_vec,shot_dur,shot_ratio,T1,T2,...
    startupechoes,tsefactor)

% Get FA train and signal of reference tissue:
[FA,mtf_xy,mtf_z] = tissueSP_sweep_shotDef(esp_vec,shot_dur,shot_ratio,...
    maxangles,startupechoes,tsefactor,...
    T1,T2); % tissue 1 (of interest)

% Signal at center of k-space for low-high profile order:
n0 = startupechoes + 1;
s0 = mtf_xy(n0);

% Calculate PSF:
s_trim = mtf_xy(startupechoes+1:end); % exclude startup echoes
psf = abs(fftshift(fft(s_trim)));

% Interpolate between points for higher accuracy:
psf = interp1(1:length(psf),psf,1:0.01:length(psf));

% Compute FWHM of PSF:
ky = -(length(psf)-1)/2:(length(psf)-1)/2; % account for zero padding
ky = ky/1000; % scale back to pixel units
fwhm = compute_fwhm(ky,psf); % no blurring => fwhm = 1
end

function width = compute_fwhm(x,y)
% Full-Width at Half-Maximum (FWHM) of the waveform y(x)
% and its polarity.
% The FWHM result in 'width' will be in units of 'x'
% Rev 1.2, April 2006 (Patrick Egan)
y = y / max(y);
N = length(y);
lev50 = 0.5;
if y(1) < lev50        % find index of center (max or min) of pulse
    [~,centerindex]=max(y);
else
    [~,centerindex]=min(y);
end
i = 2;
while sign(y(i)-lev50) == sign(y(i-1)-lev50)
    i = i+1;
end                         %first crossing is between v(i-1) & v(i)
interp = (lev50-y(i-1)) / (y(i)-y(i-1));
tlead = x(i-1) + interp*(x(i)-x(i-1));
i = centerindex+1;            %start search for next crossing at center
while ((sign(y(i)-lev50) == sign(y(i-1)-lev50)) && (i <= N-1))
    i = i+1;
end
if i ~= N
    interp = (lev50-y(i-1)) / (y(i)-y(i-1));
    ttrail = x(i-1) + interp*(x(i)-x(i-1));
    width = ttrail - tlead;
else
    width = NaN;
end
end


function [FA,s,m_long] = tissueSP_sweep_shotDef(esp_vec,shot_dur,shot_ratio,maxangles,startupechoes,tsefactor,...
    T1,T2)
plat_dur_req = shot_dur*shot_ratio;
alpha1 = min(maxangles);
left = 10;
right = alpha1;
left_limited = false;
right_limited = false;
esp_first = esp_vec(1);
esp = esp_vec(2);
% Calculation of plateau duration for a first estimated value of a_min:
a_min = left;
% Interpolate asymptotically between a first 180 angle and a_min using
% tabulated values:
FA_pss = pss_sweep_table(a_min); % always returns 1 + 5 angles
% Get FA train and signal:
[FA,s,n_max1,m_long] = epg_inverse(esp_vec,maxangles,startupechoes,FA_pss,...
    tsefactor,T1,T2);
% Check if left-limited:
plateau_dur_calc = esp_first + esp*(n_max1-1);
if plateau_dur_calc < plat_dur_req
    left_limited = true;
%     a_min_result = left; % use this value of a_min
end
% If NOT left-limited, calculate alpha_cent and plateau duration for a second
% estimated value of a_min:
if ~left_limited
    a_min = right;
    FA_pss = pss_sweep_table(a_min);
    [FA,s,n_max1,m_long] = epg_inverse(esp_vec,maxangles,startupechoes,FA_pss,...
        tsefactor,T1,T2);
    plateau_dur_calc = esp_first + esp*(n_max1-1);
        if plateau_dur_calc >= plat_dur_req
            right_limited = true;
%             a_min_result = right;
        end
end
% If neither left- or right-limited, find a_min using bisection
% algorithm:
if ~left_limited && ~right_limited
    req_dur = plat_dur_req;
    [~,FA,s,~,m_long] = find_a_min(req_dur,esp_vec,...
    maxangles,left,right,startupechoes,T1,T2,tsefactor);
end
end


function FA_pss = pss_sweep_table(a_min)
% Find flip angle sweep corresponding to a pseudo steady state
% (PSS) transition between a maximum angle and a specified minimum angle.
% Output:
%   FA_pss = PSS sweep from 180 to a_min

M = 20;
N = 5;
a = [ 120.0,  58.0,  37.0,  25.0,  10.0;...
        120.0,  58.0,  37.0,  25.0,  15.0;...
        125.0,  62.0,  38.0,  28.0,  20.0;...
        125.0,  65.0,  43.0,  34.0,  25.0;...
        125.0,  66.0,  44.0,  36.0,  30.0;...
        128.0,  70.0,  50.0,  40.0,  40.0;...
        134.0,  78.0,  57.0,  50.0,  50.0;...
        138.0,  86.0,  66.0,  60.0,  60.0;...
        144.0,  96.0,  75.0,  70.0,  70.0;...
        149.0, 105.0,  85.0,  80.0,  80.0;...
        150.0, 110.0,  94.0,  90.0,  90.0;...
        151.0, 112.0, 100.0, 100.0, 100.0;...
        156.0, 121.0, 110.0, 110.0, 110.0;...
        156.0, 127.0, 120.0, 120.0, 120.0;...
        157.0, 133.0, 130.0, 130.0, 130.0;...
        160.0, 141.0, 140.0, 140.0, 140.0;...
        165.0, 150.0, 150.0, 150.0, 150.0;...
        170.0, 160.0, 160.0, 160.0, 160.0;...
        175.0, 170.0, 170.0, 170.0, 170.0;...
        180.0, 180.0, 180.0, 180.0, 180.0];    
eff_nom_ratio = 1.0;
a_eff = max( eff_nom_ratio * a_min, 10.0 );
% Find m such that we can interpolate linearly between sweeps m and m+1.
% Make sure that m >= 1 and m < M:
m = 1;
spss_angles = zeros(1,N);
while a(m+1,N) <= a_eff && m < M-1
    m = m+1;
end
m_frac = (a_eff - a(m,N))/(a(m+1,N) - a(m,N));
% Do the interpolation for N pulses:
for n=1:N
    a_intpl = a(m,n) + m_frac*(a(m+1,n) - a(m,n));
    spss_angles(n) = min(a_intpl/eff_nom_ratio,180);
end
FA_pss = [180 spss_angles]; % include an initial 180 pulse
end

function [s,m_long,FZall] = epg_forward(flipangle,numberechos,esp_vec,T1,T2,m0_frac)
% Forward EPG algorithm for computing signal values
% Output:
%   s = signal (real-valued)
%   FZall = array containing F and Z states

if (isempty(numberechos))
    numberechos = length(flipangle);
elseif (length(flipangle)==1 && numberechos>1 && abs(flipangle)<pi)
    flipangle(2) = flipangle(1);
    flipangle(1) = (pi*exp(1i*angle(flipangle(2)))+flipangle(2))/2;
% (Henning's 1st flip reduced trick)
elseif (numberechos>length(flipangle))
    flipangle(end+1:numberechos) = flipangle(end);
end
% Convert degrees to radians:
flipangle = flipangle*pi/180;
% Allocate all known states:
FZ = zeros(3,2*numberechos);
% Prepare arrays to store all F and Z states:
FZall = cell(1,1,numberechos); % Add dephased states with each gradient application
% Prepare array to store signal:
s = zeros(1,numberechos);
m_long = zeros(1,numberechos); % longitudinal magnetisation
% Initial conditions after excitation:
FZ(1,1) = 1*m0_frac;
FZ(2,1) = 1*m0_frac;
% Refocusing pulses and dephasing:
for ech=1:numberechos
    FZ = epg_dephase(FZ,esp_vec(ech)/2,1,T1,T2); % Left crusher
    FZ = epg_rf(FZ,flipangle(ech)); % Refocusing RF pulse
    FZ = epg_dephase(FZ,esp_vec(ech)/2,1,T1,T2); % Right crusher
    % Store signal corresponding to current echo:
    s(ech) = abs(FZ(1,1));
    m_long(ech) = abs(FZ(3,1));
    % Store states:
    FZall{1,1,ech} = FZ;
end
end

function FpFnZ = epg_dephase(FpFnZ,t,Gon,T1,T2)
% Propagate EPG states through a period of dephasing due to relaxation and
% gradients. Describes time evolution between RF pulses
% Output:
%   FpFnZ = updated F+, F- and Z states

% Decay of states due to relaxation alone:
E1 = exp(-t/T1);
E2 = exp(-t/T2);
EE = diag([E2 E2 E1]);
RR = 1-E1; % Mz recovery, affects only Z0 state
% Relaxation (and recovery):
FpFnZ = EE * FpFnZ;
FpFnZ(3,1) = FpFnZ(3,1) + RR;
% Advance states:
if Gon
    FpFnZ = epg_grad(FpFnZ);
end
end

function FpFnZ = epg_grad(FpFnZ)
% Propagate states through a "unit" gradient
% Output:
%   FpFnZ = Updated FpFnZ state

% Gradient shifts dephasing order of F states:
FpFnZ = [FpFnZ zeros(3,20)]; % Add higher dephased states
FpFnZ(1,:) = circshift(FpFnZ(1,:),[0 1]); % Shift Fp states
FpFnZ(2,:) = circshift(FpFnZ(2,:),[0 -1]); % Shift Fn states
FpFnZ(2,end) = 0; % Zero highest Fn state
FpFnZ(1,1) = conj(FpFnZ(2,1)); % Fill in lowest Fp state
end

function FpFnZ = epg_rf(FpFnZ,alpha)
% Propagate EPG states through an RF pulse
% Input:
%   FpFnZ = 3xN vector of F+, F- and Z states
%   alpha = flip angle of pulse (radians)
% Output:
%   FpFnZ = Updated FpFnZ state

if (real(alpha) > 2*pi)
    warning('epg_rf:  Flip angle should be in radians!')
end
T = [(cos(alpha/2))^2 (sin(alpha/2))^2 sin(alpha);
    (sin(alpha/2))^2 (cos(alpha/2))^2 -sin(alpha);
    -0.5*sin(alpha) 0.5*sin(alpha) cos(alpha)];
% T mixes states of equal dephasing order:
FpFnZ = T * FpFnZ;
end

function [FA,s,n_max1,m_long] = epg_inverse(esp_vec,a_max,N_startup,FA_pss,...
    TSEfactor,T1,T2)
% Inverse EPG algorithm used to compute flip angles corresponding to given
% signal values
% Output:
%   FA = array of computed flip angles
%   s = array of corresponding signal values
%   n_max1 = RF pulse at which min(a_max) is reached

esp = esp_vec(2);
E2 = exp(-0.5*esp/T2);
% s = s_spss;
alpha_max1 = a_max(1)*pi/180;
alpha_max2 = a_max(2)*pi/180;
max1_reached = false;
FZ = zeros(3,2*length(esp_vec)); % array containing states
FA = zeros(1,length(esp_vec));
s = zeros(1,length(esp_vec));
m_long = zeros(1,length(esp_vec));
% Initial conditions after excitation:
FZ(1,1) = 1;
FZ(2,1) = 1;
% n = 1;
% while length(s) < TSEfactor+N_startup+1
for n=1:TSEfactor+N_startup+2 % extra "+1" for looping purposes
    if n <= 6 % only calculate FA for pulses after the startup echoes
        FA(n) = FA_pss(n)/180*pi; % in radians
        FZ = epg_dephase(FZ,esp_vec(n)/2,1,T1,T2); % relaxation + left crusher
        FZ = epg_rf(FZ,FA(n)); % RF pulse with current FA
        FZ = epg_dephase(FZ,esp_vec(n)/2,1,T1,T2); % right crusher + relaxation
    else
        FA_act = FA(1:n-1); % in radians
        [s,m_long] = epg_forward(FA_act*180/pi,length(FA_act),...
            esp_vec(1:length(FA_act)),T1,T2,1);
        s_curr = s(end);
        FZ = epg_dephase(FZ,esp/2,1,T1,T2); % relaxation + left crusher
        % Define parameters to be used in calculation:
        A = FZ(1,2) - s_curr/E2; % FZ(1,2) is F1 state prior to pulse
        B = FZ(2,2) - s_curr/E2; % FZ(2,2) is F-1 state prior to pulse
        Z1 = FZ(3,2); % Z1 state prior to pulse
        root_arg = Z1^2 - A*B;
        % Calculate flip angle:
        if abs(A) > 1e-10
            if root_arg >= 0
                FA_out = 2*atan((Z1 + sqrt(root_arg))/A);
            else
                FA_out = 2*pi; % no solution, apply maximum angle
            end
        else
            if abs(Z1) > 1e-10
                FA_out = 2*atan(0.5*B/Z1);
            else
                FA_out = 2*pi; % no solution, apply maximum angle
            end
        end
        % Make sure 0 < FA <= 180:
        if FA_out > pi
            FA_out = FA_out - pi;
        elseif FA_out < 0
            FA_out = FA_out + pi;
        end
        % Clip FA if larger than alpha_max1 and continue with linear sweep from
        % alpha_max1 to alpha_max2:
        if FA_out >= alpha_max1 && min(FA) <= alpha_max1
            if ~max1_reached
                max1_reached = true;
                n_max1 = n;
            end
        else
            n_max1 = length(esp_vec); % if alpha_max1 not reached, return last
        end
        if max1_reached
            if n < length(esp_vec) % length(s) = # angles = N_init+1+etl
            slope = (alpha_max2-alpha_max1)/(length(esp_vec)-n_max1);
            FA_out = alpha_max1 + slope*(n-n_max1);
            else
            FA_out = alpha_max2;
            end
        end
        if n == TSEfactor+N_startup+2
            break
        end
        % Continue with sequence:
        FZ = epg_rf(FZ,FA_out); % RF pulse with current FA
        FZ = epg_dephase(FZ,esp/2,1,T1,T2); % right crusher + relaxation
        FA(n) = FA_out; % in radians
    end
%     n=n+1;   
end
fclose('all'); % close all open files
FA = FA*180/pi; % convert back to degrees
FA=FA';
s=s';
m_long = m_long';
end

function [a_min_result,FA,s,n_max1,m_long] = find_a_min(req_dur,esp_vec,...
    a_max,left,right,N_startup,T1,T2,TSEfactor)
iteration = 0;
plateau_dur_calc = 0; % just to get while loop executed at least once
esp_first = esp_vec(1);
esp = esp_vec(2);
while iteration < 12 && abs(plateau_dur_calc-req_dur) > esp/2
    mid = (left+right)/2;
    a_min = mid; % Calculate plateau duration for a_min = mid:
    FA_pss = pss_sweep_table(a_min);
    [FA,s,n_max1,m_long] = epg_inverse(esp_vec,a_max,N_startup,FA_pss,...
    TSEfactor,T1,T2);
    plateau_dur_calc = esp_first + esp*(n_max1-1);
    if plateau_dur_calc > req_dur
        left = mid;
    else
        right = mid;
    end
    a_min_result = mid;
    iteration = iteration +1;
end
end
