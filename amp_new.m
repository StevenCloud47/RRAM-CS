%% AMP single test with noise flow 2024-3-30
clear all;
% read pic
h = im2double(imread('.\pre-pic\bird.png'));
N = 256;
% parameters
cr = 0.75;
orth_flag = 1;
num = 100; 
epsilon = 1e-6;
alpha = 1.0;
handle_2d = 0;
% read_level = nl; % typical, or 1% (optimized)
% map_level = nl; % typical, or 1% (optimized)
choice = [0,0,0,1];
fir1 = [6,0.75,0,0];
fir2 = [4,0.9,0,0];
x_fir = ones(N,1);
trans_type = 'dct';
show_detail = 0;

%[PSNR,hat_theta,hat_h] = amp_2d_gray(N,h,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);

PSNRs = [];
PSNRs_ol = [];
begins = 0.00;
ends = 0.20;
steps = 0.01;
c = find(choice == 1);

for nl = begins:steps:ends
    read_level = nl; % typical, or 1% (optimized)
    map_level = nl; % typical, or 1% (optimized)
    tic;
    [PSNR,~,~] = amp_2d_gray(N,h,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    [PSNR_ol,~,~] = amp_2d_gray(N,h,cr,0,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    toc;
    PSNRs = [PSNRs,PSNR(c)];
    PSNRs_ol = [PSNRs_ol,PSNR_ol(c)];
    fprintf('Now progress : %d in %d, PSNR = %2f,PSNR_ol = %2f\n',nl/steps+1,ends/steps+1,PSNRs(end),PSNRs_ol(end));
end

% above experiment plot 2024-4-1 
nls = begins:steps:ends;
figure(1);
plot(nls,PSNRs,'-*','linewidth',1.5);
grid;
xlim([0 0.2]); 
xticks(0:0.02:0.2);
ylim([0 30]);
xlabel('noise level','Fontsize',12);
ylabel('PSNR','Fontsize',12);
title('AMP reconstruction PSNR-noise level curve','Fontsize',12);

figure(2);
plot(nls,PSNRs,'-*','linewidth',1.5);
hold on;
plot(nls,PSNRs_ol,'-*','linewidth',1.5);
grid;
xlim([0 0.2]); 
xticks(0:0.02:0.2);
ylim([0 30]);
xlabel('noise level','Fontsize',12);
ylabel('PSNR','Fontsize',12);
title('AMP reconstruction PSNR-noise level curve','Fontsize',12);
legend('orth','orthless','Fontsize',10);


%% dataset reconstruction with noise flow 2024-4-1
% save params to params_1.mat
clear all;
load ('.\save-mat\params_0401.mat');
choice = [0,0,0,1];
source_root = '.\ImageNet\0401-1class\val-100k-rgb\';
target_root = '.\ImageNet\0401-1class\recons-val-100k\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 1;
save_flag = 0;

PSNR_mu = [];
PSNR_sigma = [];
begins = 0.00;
ends = 0.20;
steps = 0.01;

for nl = begins:steps:ends
    read_level = nl; % typical, or 1% (optimized)
    map_level = nl; % typical, or 1% (optimized)
    PSNRs = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
    [mu_now,sigma_now,~] = PSNR_evaluate(PSNRs);
    PSNR_mu = [PSNR_mu,mu_now];
    PSNR_sigma = [PSNR_sigma,sigma_now];
end

%%  above experiment plot 2024-4-1 
load('.\save-mat\psnr_0401_noise_scan.mat');
nls = begins:steps:ends;
figure(1);
plot(nls,PSNR_mu,'-*','linewidth',1.5);
grid;
xlim([0 0.2]); 
xticks(0:0.02:0.2);
ylim([0 35]);
xlabel('noise level','Fontsize',12);
ylabel('avg-PSNR','Fontsize',12);
title('AMP reconstruction PSNR-noise level curve','Fontsize',12);

%save('.\save-mat\psnr_0401_noise_scan.mat');


%% dataset reconstruction with another 5 class 2024-4-1
clear all;
load ('.\save-mat\params_0401.mat');
source_root = '.\ImageNet\0402-5class\val-100k-rgb\';
target_root = '.\ImageNet\0402-5class\recons-val-100k-rgb\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = length(class_name);
save_flag = 1;

choice = [1,0,0,0];
PSNRs = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
               read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
[mu,sigma,PSNR_vec] = PSNR_evaluate(PSNRs);

choice = [0,0,0,1];
target_root = '.\ImageNet\0402-5class\recons-val-100k-rgb-noise\';
PSNRs_noise = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
               read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
[mu_noise,sigma_noise,PSNR_vec_noise] = PSNR_evaluate(PSNRs_noise);

PSNR_stat(PSNR_vec,PSNR_vec_noise);
save('.\save-mat\psnr_0402_5class.mat');


%% dataset reconstruction with orth and orthless 2024-4-2
% save params to params_1.mat
clear all;
load ('.\save-mat\params_0401.mat');
source_root = '.\ImageNet\0321-5class\val-100k\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-noise\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = length(class_name);
save_flag = 0;

orth_flag = 0;
choice = [1,0,0,0];
PSNRs_orthless = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
               read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
[mu_orthless,sigma_orthless,PSNR_vec_orthless] = PSNR_evaluate(PSNRs_orthless);

choice = [0,0,0,1];
PSNRs_orthless_noise = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
               read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
[mu_orthless_noise,sigma_orthless_noise,PSNR_vec_orthless_noise] = PSNR_evaluate(PSNRs_orthless_noise);

save('.\save-mat\psnr_0402_orthless.mat');
PSNR_stat(PSNR_vec_orthless,PSNR_vec_orthless_noise);

%% single image test with compression_rate&noise_level sweep
clear all;
h = im2double(imread('.\pre-pic\bird.png'));
load ('.\save-mat\params_0401.mat');
choice = [0,0,0,1];
c = find(choice == 1);
PSNRs = [];
PSNRs_nl = [];

cr_begins = 0.4;
cr_ends = 1.0;
cr_steps = 0.05;
crs = cr_begins:cr_steps:cr_ends;
nl_begins = 0.00;
nl_ends = 0.20;
nl_steps = 0.01;
nls = nl_begins:nl_steps:nl_ends;
PSNRs_c_n = zeros(length(crs),length(nls));
for i = 1:1:length(crs)
    cr_now = crs(i);
    for j = 1:1:length(nls)
        map_level = nls(j);
        read_level = nls(j);
        tic;
        [PSNRs_now,~,~] = amp_2d_gray(N,h,cr_now,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
        PSNRs_c_n(i,j) = PSNRs_now(c);
        fprintf('Now progress : cr = %2f,nl = %2f, PSNR = %2f dB,',crs(i),nls(j),PSNRs_c_n(i,j));
        toc;
    end
end


%%  above experiment plot 2024-4-2
%save('.\save-mat\psnr_0402_cr_nl.mat');
figure(1);
grid;
hold on;
for i = 1:length(crs)
    if(crs(i)>0.49 && crs(i)<0.76)
        %plot(nls, PSNRs_c_n(i,:),  'DisplayName', sprintf('cr=%2f',crs(i)),'-*','linewidth',1.2);
        plot(nls, PSNRs_c_n(i,:),'-*','DisplayName', sprintf('cr = %.2f', crs(i)),'linewidth',1.0);
    end
end
legend show;
xlim([0 0.2]); 
xticks(0:0.02:0.2);
ylim([0 30]);
xlabel('noise level','Fontsize',12);
ylabel('PSNR','Fontsize',12);
title('AMP reconstruction PSNR-noise level&compression ratio','Fontsize',12);


%% dataset reconstruction with noise flow 2024-4-2~4-6
% save params to params_1.mat
clear all;
load ('.\save-mat\params_0401.mat');
choice = [0,0,0,1];
source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
class_name = getAllSubfolders(source_root);
begin_class = 4;
end_class = 5;
save_flag = 1;

begins = 0.11;
ends = 0.15;
steps = 0.01;
nls = begins:steps:ends;

PSNR_mu = zeros(1,length(nls));
PSNR_sigma = zeros(1,length(nls));

for i = 1:1:length(nls)
    target_root = ['.\ImageNet\0321-5class\recons-val-100k-rgb-noise-',num2str(nls(i)),'\'];
    read_level = nls(i); % typical, or 1% (optimized)
    map_level = nls(i); % typical, or 1% (optimized)
    PSNRs = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);
    [mu_now,sigma_now,~] = PSNR_evaluate(PSNRs);
    PSNR_mu(i) = mu_now;
    PSNR_sigma(i) = sigma_now;
end

%save('.\save-mat\psnr_0403_noise_scan_6.mat');
% 0403:_1:0~0.04 _2:0.05~0.06 _3:0.07~0.08 _4:0.09~0.11 _5:0.12~0.13 _6:0.14~0.15

save('.\save-mat\psnr_0404_noise_scan_3.mat');
% 0404:_1:0~0.06 _2:0.07~0.10 _3:0.11~0.15


%% plot for above: PSNR-noise
figure(1);
ends_now = 0.15;
PSNR_mu = xlsread('noise_level.xlsx','C15:R15');
% ends_now = 0.10;
% PSNR_mu = xlsread('noise_level.xlsx','C15:M15');
nls_now = 0:0.01:ends_now;


plot(nls_now,PSNR_mu,'-*','linewidth',1.5);
grid;
xlim([-0.01 ends_now+0.01]); 
xticks(0:0.03:ends_now);
ylim([10 32]);
xlabel('noise level','Fontsize',12);
ylabel('avg-PSNR','Fontsize',12);
title('AMP reconstruction PSNR-noise level curve','Fontsize',12);


%% plot for above: acc-noise，PSNR-noise，acc-PSNR
clear all;
close all;
figure(1);
ends_now = 0.09;
nls_now = 0:0.01:ends_now;
nets = {'ResNet18','ResNet34','ResNet50','ResNet101','ResNet152'};
%results = zeros(length(nets),length(nls_now)+1);

results = xlsread('noise_level.xlsx','C20:L22');
PSNR_mu = xlsread('noise_level.xlsx','C15:L15');

hold on;
for i=1:1:3
    plot(nls_now,results(i,:),'-*','linewidth',1.2,'DisplayName', sprintf(nets{i}));
end

hold off;
grid;
legend show;
xlim([0 ends_now+0.0]); 
xticks(0:0.01:ends_now+0.0);
xlabel('noise level','Fontsize',12);
ylabel('classification accuracy','Fontsize',12);
title('AMP reconstruction accuracy-noise level curve','Fontsize',12);

figure(2)
hold on;
for i=1:1:3
    plot(PSNR_mu,results(i,:),'-*','linewidth',1.2,'DisplayName', sprintf(nets{i}));
end

x_tick = 16:2:32;
% xuxian1 = 0.98 * ones(1,length(x_tick));
% xuxian2 = 0.90 * ones(1,length(x_tick));
% plot(x_tick,xuxian1,'--','linewidth',1.0);
% plot(x_tick,xuxian2,'--','linewidth',1.0);

grid;
legend show;
hold off;

% xlim([0 ends_now]); 
% xticks(0:0.01:ends_now);
xlim([16 32]);
xticks(x_tick);
xlabel('PSNR/dB','Fontsize',12);
ylabel('classification accuracy','Fontsize',12);
title('AMP reconstruction accuracy-PSNR curve','Fontsize',12);


%% 


%% gauss+orth
rng(11637);
N = 256;
cr = 0.75;
M = cr * N;
Phi = randn(M,N); 
Phi = sqrt(1/M) * Phi;
Phi = orth(Phi')';


%% ====================== Function Below ====================== %%
%% amp_eta_t 2024-2-2
function eta = amp_eta_t(r,lambda)
    %[a,b] = size(r);
    %scalar function eta(r,l) = sgn(r) * max(|r|-l,0)
    sgn1 = sign(r);
    minus1 = abs(r) - lambda;
    minus1(minus1<0) = 0;
    eta = sgn1 .* minus1;
end

%% pic2double 2024-2-2
% transform any picture h to N*N double matrix(gray picture)
function [h_double] = pic2double(h,N)
    [N1,N2,color] = size(h);
    if(color==3)
        h1 = rgb2gray(h);
    else
        h1 = h;
    end
    h2 = im2double(h1);
    
    if((N1~=N)&&(N2~=N))
        h_double = imresize(h2,[N,N]);
    else
        h_double = h2;
    end
end

%% fir_self 2024-4-1
function [b,filtered] = fir_self(a,fir_i)
    %a:original signal
    %N:stage
    %fc:cut off frequency in (0,1),1 for nyquist
    %fir1(n,fc,'ftype',window),ftype=high to design high-pass,
    %window default=hamming
    N = fir_i(1);
    fc = fir_i(2);
    type = fir_i(3);
    flag = fir_i(4);
    
    if(flag == 0)
        filtered = a;
        b = 1;
    else
        if(type == 0)
            b = fir1(N,fc,rectwin(N+1));
        elseif(type == 1)
            b = fir1(N,fc,hamming(N+1));
        elseif(type == 2)
            b = fir1(N,fc,hann(N+1));
        elseif(type == 3)
            b = fir1(N,fc,blackman(N+1));
        elseif(type == 4)
            b = fir1(N,fc,bartlett(N+1));
        elseif(type == 5)
            b = fir1(N,fc,kaiser(N+1));
        elseif(type == 6)
            b = fir1(N,fc,chebwin(N+1));
        else
            error('This type has not been developed.');
        end
        
        % filter handle
        filtered = filter(b,1,a);
        
        % phase change
        [n,~] = size(a);
        if(n>1)
            filtered = [filtered(N/2+1:end);a(end-N/2+1:end)];
        else
            filtered = [filtered(N/2+1:end),a(end-N/2+1:end)];
        end
    end
end

%% fetch all subfolders 2024-3-20
function foldersCell = getAllSubfolders(folderPath)
    % 获取指定路径下的所有内容
    items = dir(folderPath);
    
    % 过滤出所有的子文件夹（排除'.'和'..'）
    folders = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));
    
    % 提取所有子文件夹的名称
    foldersNames = {folders.name};
    
    % 将文件夹名称保存到cell数组中
    foldersCell = foldersNames;
end

%% amp_core with trans-domain measurement 2024-3-30
function [x,z] = amp_core_pre(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,noise_info,stop_earlier,show_detail)
    %y: M*1 signal after measurement
    %A: Phi * Psi
    %num: max iteration num
    %epsilon: min iteration step
    %alpha: usually alpha=1 for this fucntion
    %noise_info: [map_noise_flag,map_level,read_noise_flag,read_level]...
    %...flag==0 for noiseless
    %theta: ground truth for sparse vector to be reconstructed]
    %stop_earlier: no need for run all num iteration...
    %...if less than epsilon, break.
    
    %x: N*1 sparse signal reconstruction result
    %z: M*1 residual(during iteration)
    %MSE/NMSE: MSE/NMSE change in the iteration process
    
    map_noise_flag = noise_info(1);
    map_noise_level = noise_info(2); % typical, or 1% (optimized)
    read_noise_flag = noise_info(3);
    read_noise_level = noise_info(4); % typical, or 1% (optimized)
    
    MSE = [];
    NMSE = [];
    
    [M,N] = size(A);
    x = zeros(N,1);
    z = zeros(M,1);
    
    % map deviation
    rng(11638);
    if map_noise_flag
        noise = max(abs(A(:)))*map_noise_level*randn(M,N);
        A_map_noise = A + noise;
    else
        A_map_noise = A;
    end
    rng('shuffle');
    
    %iteration
    for t = 1:1:num
        x0 = x;
        z0 = z;
        b = sum(x(:)~=0) / M;   %AMP append element
        
        % read deviation
        if read_noise_flag
            noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
        end
        
        z = y - z1 + b*z0;
    
        lambda = alpha*norm(z,2) / sqrt(M);
        
        r1 = A_noise'*z;
        
        r = r1 + x;
            
        x = amp_eta_t(r,lambda);
        
        %recording MSE and NMSE change
        MSE_now = (norm(x - theta)^2)/N;
        NMSE_now = (norm(x - theta)^2)/(norm(theta)^2);
        MSE = [MSE,MSE_now];
        NMSE = [NMSE,NMSE_now];
        epsilon_now = norm((x - x0),2) / (norm(x,2)+1e-8);
        
        % show detail
        if show_detail == 1
            fprintf('进行第%d次迭代,epsilon = %4f \n',t,epsilon_now);
        end
        
        % stop earlier
        if(stop_earlier==1)
            if(epsilon_now < epsilon)       
                if show_detail == 1
                    fprintf('在%d次迭代后收敛,结束循环\n',t); 
                end
                break;
            end
        end     
    end
    
    %变换域滤波
    x = x.*x_fir;    
end

% 1D-AMP for sparse vector (new_version)
function [x,z] = amp_core(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,noise_info,stop_earlier,show_detail)
    %y: M*1 signal after measurement
    %A: Phi * Psi
    %num: max iteration num
    %epsilon: min iteration step
    %alpha: usually alpha=1 for this fucntion
    %noise_info: [map_noise_flag,map_level,read_noise_flag,read_level]...
    %...flag==0 for noiseless
    %theta: ground truth for sparse vector to be reconstructed]
    %stop_earlier: no need for run all num iteration...
    %...if less than epsilon, break.
    
    %x: N*1 sparse signal reconstruction result
    %z: M*1 residual(during iteration)
    %MSE/NMSE: MSE/NMSE change in the iteration process
    
    map_noise_flag = noise_info(1);
    map_noise_level = noise_info(2); % typical, or 1% (optimized)
    read_noise_flag = noise_info(3);
    read_noise_level = noise_info(4); % typical, or 1% (optimized)
    
    MSE = [];
    NMSE = [];
    
    [M,N] = size(A);
    x0 = zeros(N,1);
    z0 = y;
    
    % map deviation
    rng(11638);
    if map_noise_flag
        noise = max(abs(A(:)))*map_noise_level*randn(M,N);
        A_map_noise = A + noise;
    else
        A_map_noise = A;
    end
    rng('shuffle');
    
    %iteration
    % x0,z0→x,z
    for t = 1:1:num
        
        lambda0 = alpha*norm(z0,2) / sqrt(M);
        
        b0 = sum(x0(:)~=0) / M;   %AMP append element        
        % read deviation
        if read_noise_flag
            noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
        end
        
        [~,r0] = fir_self(A_noise' * z0, fir1);
        
        x = amp_eta_t(r0 + x0, lambda0);
        
        [~,s] = fir_self(A_noise * x, fir2);
        
        z = y - s + z0 * b0;
        
        %recording MSE and NMSE change
        MSE_now = (norm(x - theta)^2)/N;
        NMSE_now = (norm(x - theta)^2)/(norm(theta)^2);
        MSE = [MSE,MSE_now];
        NMSE = [NMSE,NMSE_now];
        epsilon_now = norm((x - x0),2) / (norm(x,2)+1e-8);
        
        % show detail
        if show_detail == 1
            fprintf('进行第%d次迭代,epsilon = %4f \n',t,epsilon_now);
        end
        
        % stop earlier
        if(epsilon_now < epsilon)
            if(stop_earlier==1)       
                if show_detail == 1
                    fprintf('在%d次迭代后收敛,结束循环\n',t); 
                end
                break;
            end
        end      
        x0 = x;
        z0 = z;
    end
    
    %变换域滤波
    x = x.*x_fir;    
end

% 2D-AMP for single real 2D map: N*N theta,compress and recovery
function [hat_theta] = amp_2d(theta,Phi,Psi,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,show_detail)
    
    %noise
    noise_info = [0,map_level,0,read_level;
                  1,map_level,0,read_level;
                  0,map_level,1,read_level;
                  1,map_level,1,read_level];
              
    %choice
    sum_choice = sum(choice);
    if((sum_choice<1)||(sum_choice>4))
        error('choice input must in range [0,0,0,1] ~ [1,1,1,1]');
    end
    
    [~,N] = size(Phi);
    hat_theta = zeros(N,N,4);
    
    for noise_tag = 1:1:4
        if(choice(noise_tag)==0)
            continue;
        else
            if handle_2d == 0
                % column-wise handle
                for i = 1:1:N
                    theta_now = theta(:,i);
                    %theta_now = theta(i,:)';
                    x = Psi * theta_now;
                    y = Phi * x;     
                    A = Phi * Psi;                    
                    [hat_theta_elem,~] = amp_core(y,A,theta_now,num,epsilon,alpha,fir1,fir2,x_fir,noise_info(noise_tag,:), ...
                                                  1,show_detail);               
                    hat_theta(:,i,noise_tag) = hat_theta_elem;
                end
            elseif handle_2d == 1
                % row-wise handle
                for i = 1:1:N
                    theta_now = theta(i,:)';
                    x = Psi * theta_now;
                    y = Phi * x;     
                    A = Phi * Psi;                    
                    [hat_theta_elem,~] = amp_core(y,A,theta_now,num,epsilon,alpha,fir1,fir2,x_fir,noise_info(noise_tag,:), ...
                                                  1,show_detail);               
                    hat_theta(:,i,noise_tag) = hat_theta_elem';
                end
            else
                error('This handle has not been developed.');
            end
        end
    end
end

% 2D-AMP for single channel picture compression and recovery
function [PSNR,hat_theta,hat_h] = amp_2d_gray(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail)
    
    h = pic2double(h,N);
    M = round(N * compress_rate); 
    
    %Gauss
    rng(11637);
    Phi = randn(M,N); 
    Phi = sqrt(1/M) * Phi;
    if orth_flag == 1
        Phi = orth(Phi')';
    end
    Psi = eye(N,N);
    
    % record result
    hat_h = zeros(N,N,4);
    PSNR = zeros(1,4);
    
    %trans_type
    if trans_type == 'dct'
        % dct sparse
        theta = dct2(h);
        hat_theta = amp_2d(theta,Phi,Psi,num,epsilon,alpha,handle_2d,...
                    map_level,read_level,choice,fir1,fir2,x_fir,show_detail);
        % record PSNR
        for noise_tag = 1:1:4
            if(choice(noise_tag)==0)
                continue;
            else
                hat_h(:,:,noise_tag) = idct2(hat_theta(:,:,noise_tag));
            end
            PSNR(noise_tag) = psnr(hat_h(:,:,noise_tag),h);     
        end
    elseif trans_type == 'non'
        % origin sparse
        theta = h;
        hat_theta = amp_2d(theta,Phi,Psi,num,epsilon,alpha,handle_2d,...
                    map_level,read_level,choice,fir1,fir2,x_fir,show_detail);
        % record PSNR
        for noise_tag = 1:1:4
            if(choice(noise_tag)==0)
                continue;
            else
                hat_h(:,:,noise_tag) = (hat_theta(:,:,noise_tag));
            end      
            PSNR(noise_tag) = psnr(hat_h(:,:,noise_tag),h);
        end
    else
        error('This trans_type has not been developed.');
    end
    
    % show PSNR detail
    formatspec_tot = cell(4,1);
    formatspec_tot{1} = 'noiseless重构PSNR = %4f \n';
    formatspec_tot{2} = 'only map noise重构PSNR = %4f \n';
    formatspec_tot{3} = 'only read noise重构PSNR = %4f \n';
    formatspec_tot{4} = 'map+read noise重构PSNR = %4f \n';
    for j=1:1:4
        if(choice(j)==1 && show_detail==1)
            fprintf(formatspec_tot{j},PSNR(j));
        end 
    end   
end

% 2D-AMP for 3-channel RGB picture compression and recovery
function [PSNR,hat_theta,hat_h] = amp_2d_rgb(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail)
    % need to be rgb
    [~,~,color] = size(h);
    if(color~=3)
        error('Picture Must be RGB three channel.')
    end
    % record result
    hat_theta = zeros(N,N,3,4);
    hat_h = zeros(N,N,3,4);
    PSNR_now = zeros(1,3,4);
    % 3 channels reconstruction
    for i = 1:1:3
        h_now = h(:,:,i);
        [PSNR_now(:,i,:),hat_theta(:,:,i,:),hat_h(:,:,i,:)] = amp_2d_gray(N,h_now,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
end

%% total_dataset_reconstruction and evaluate[gray or rgb] 2024-3-30
% dataset reconstruction
function [PSNRs] = amp_2d_recons(N,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag)
     
    PSNRs = cell(1,end_class - begin_class + 1);    
    for i = begin_class:1:end_class
        tic;
        class = [(class_name{i}),'\'];
        %create target folder
        if ~(isfolder([target_root,class]))
            disp('target folder doesnt exist,create it.');
            mkdir([target_root,class]);
            disp('successfully create.')
        end
        
        images = dir(fullfile(source_root,class,'*.png'));
        images_num = length(images);
        
        c_indexes = find(choice == 1);
        c_index = c_indexes(1);
        disp(c_index);
        
        i_real = i - begin_class + 1;
        PSNRs{i_real} = [];
        for j = 1:1:images_num
            % read
            pic_name = images(j).name;   
            h = imread([source_root,class,pic_name]);
            [~,~,color] = size(h);
            % recons and save
            if color == 1
                [PSNR,~,hat_h] = amp_2d_gray(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
                if save_flag == 1                   
                    imwrite(hat_h(:,:,c_index),[target_root,class,pic_name]);
                end                
            elseif color == 3
                [PSNR,~,hat_h] = amp_2d_rgb(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
                if save_flag == 1                   
                    imwrite(hat_h(:,:,:,c_index),[target_root,class,pic_name]);
                end
            else
                error('Channels must be 1 or 3');
            end
            % print
            fprintf('class %s recons-progress : %d in %d ,PSNR = %2f,class: %d in %d\n',class_name{i},j,images_num,PSNR(c_index),i-begin_class,end_class-begin_class);
            PSNRs{i_real} = [PSNRs{i_real},PSNR(c_index)];
        end
        toc;
    end
end

% cell2vector and evaluate PSNR
function [mus,sigmas,PSNR_vec] = PSNR_evaluate(PSNRs)
    PSNR_vec = [];
    class_num = length(PSNRs);
    % cell to vector
    for i=1:1:class_num
        PSNR_vec = [PSNR_vec,PSNRs{i}];
    end
    mus = mean(PSNR_vec);
    sigmas = std(PSNR_vec);
end

% evaluate statistical
function [] = PSNR_stat(PSNR_vec,PSNR_vec_noise)
   
    N_bins = 50;
    fontsize = 15;
    labelsize = 13;

    figure;
    histogram(PSNR_vec,N_bins,'FaceColor',"#EDB120");
    xlim([10,50]);
    xlabel('PSNR/dB','FontSize',labelsize);ylabel('frequency','FontSize',labelsize);

    hold on;
    histogram(PSNR_vec_noise,N_bins,'FaceColor',"#0072BD");
    title('PSNR distribution','FontSize',fontsize);
    legend('noiseless','map+read noise','FontSize',10);
    grid;
end

%% coherence between 2 matrix 2024-4-2


