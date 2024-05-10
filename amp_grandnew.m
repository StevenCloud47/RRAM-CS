%% 2024-4-24 1d test new 
clear all;
close all;
load ('.\save-mat\params_0504.mat');
% % fir0 = [2,0.8,1,0];
% % fir1 = [6,0.75,1,0];
% % fir2 = [4,0.9,1,0];
% meas_noise_flag = 1;
% % save('.\save-mat\params_0424.mat');
% x = im2double(imread('.\pre-pic\bird.png'));
% x_now = x(:,1);
% choice = 4;
% crs = 0.4:0.05:0.9;
% NMSEs = zeros(length(crs),1);
% tic;
% for i = 1:1:length(crs)
%     cr = crs(i);
%     M = round(N * cr);
%     Phi = meas_matrix_generate(M,N,16,meas_type,orth_flag);
%     Psi = trans_matrix_generate(N,trans_type);
% 
%     [hat_x,h] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info(choice,:),meas_noise_flag,stop_earlier,show_detail);
%     NMSE = (norm(x_now - hat_x)^2)/(norm(x_now)^2);
%     NMSEs(i) = NMSE;
% end
% toc;
% 
% filename = 'noise_level.xlsx';
% % 指定起始单元格位置
% startCell = 'B5';
% if choice == 4
%     startCell = 'C5';
% end
% % 写入数据
% tic
% writematrix(NMSEs, filename, 'Sheet', '2', 'Range', startCell);
% toc


%% 2024-4-24 2d test1:single picture 
% clear all;
% close all;
% load ('.\save-mat\params_0504.mat');
% choice = 4;
% handle_2d = 0;
% meas_noise_flag = 0;
% M = round(N * cr);
% Phi = meas_matrix_generate(M,N,16,meas_type,orth_flag);
% Psi = trans_matrix_generate(N,trans_type);
% 
% x = im2double(imread('.\pre-pic\1.png'));
% x = imresize(x,[N,N]);
% 
% tic;
% map_level = 0.05;
% read_level = 0.05;
% [~,~,color] = size(x);
% if(color==1)
%     [PSNR,hat_x] = amp_2d(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
% else
%     [PSNR,hat_x] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
% end
% toc;


%% 2024-5-6 参数保存与单张图像仿真
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
% seed = 11637;
% tic;
% [Gauss_noise_map_NN,Gauss_noise_read_NN] = iters_generate(N,N,seed);
% toc;
% % 
% % fir3 = [4,0.7000,1,0];
% save('.\save-mat\params_0504.mat');
choice = 4;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

x = im2double(imread('.\pre-pic\6.png'));
x = imresize(x,[N,N]);

map_level = 0.02;
read_level = 0.02;

tic;
[PSNR,hat_x_rgb] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN);
toc;

%% 2024-5-7 单张图像PSNR随noise level变化
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
choice = 4;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

x = im2double(imread('.\pre-pic\6.png'));
x = imresize(x,[N,N]);

limits = 0:0.005:0.1;
L = length(limits);
PSNRs = zeros(1,L);
hat_x_rgb_set = zeros(N,N,3,L);

tic;
for i = 1:1:L
    map_level = limits(i);
    read_level = limits(i);
    
    [PSNR_now,hat_x_rgb_now] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                    handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN);
  
    PSNRs(i) = PSNR_now;
    hat_x_rgb_set(:,:,:,i) = hat_x_rgb_now;
    fprintf('progress %d in %d, PSNR = %.2f dB\n',i,L,PSNR_now);
end
toc;

%_1-only meas _2-only recons _3-only inv-trans
PSNRs_3 = PSNRs; hat_x_rgb_set_3 = hat_x_rgb_set;

save('.\save-mat\psnr_0508_singlepic_3.mat','PSNRs_3','hat_x_rgb_set_3');

%% 2024-5-8 上个实验的画图
clear all;
close all;

limits = 0:0.005:0.1;

load('.\save-mat\psnr_0508_singlepic.mat');
load('.\save-mat\psnr_0508_singlepic_1.mat');
load('.\save-mat\psnr_0508_singlepic_2.mat');
load('.\save-mat\psnr_0508_singlepic_3.mat');
figure;

plot(limits,PSNRs,'-*','LineWidth',1.2);
hold on;
plot(limits,PSNRs_1,'-s','LineWidth',1.2);
plot(limits,PSNRs_2,'-o','LineWidth',1.2);
plot(limits,PSNRs_3,'-x','LineWidth',1.2);
xlabel('noise level','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
grid;
xlim([limits(1)-0.005 limits(end)+0.005]);
xticks(limits(1):0.01:limits(end))
title('single picture PSNR-noise level curve','FontSize',15);
legend('all noise','only meas noise','only recons noise','only inv-trans noise','FontSize',10);


%% 2024-5-8 5类数据集重建：无噪与带噪
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
choice = 1; %4为带噪
nls = 0.02;
map_level = nls;
read_level = nls;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
target_root = '.\ImageNet\0321-5class\new-frame\recons-val-100k-rgb-0508-noiseless\';
% target_root = ['.\ImageNet\0321-5class\new-frame\recons-val-100k-rgb-0508-noise-',num2str(nls),'\'];
% source_root = '.\ImageNet\selected\original\';
% target_root = ['.\ImageNet\selected\recons-new-frame-0508-noise-',num2str(nls),'\'];

class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;

[PSNRs_noiseless] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);


[mus_noiseless,sigmas_noiseless,PSNR_vec_noiseless] = PSNR_evaluate(PSNRs_noiseless);

choice = 4; %4为带噪
target_root = ['.\ImageNet\0321-5class\new-frame\recons-val-100k-rgb-0508-noise-',num2str(nls),'\'];
[PSNRs] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);


[mus,sigmas,PSNR_vec] = PSNR_evaluate(PSNRs);

save('.\save-mat\psnr_0508_5class_new_frame.mat','mus_noiseless','sigmas_noiseless','PSNR_vec_noiseless','mus','sigmas','PSNR_vec');


%% 2024-5-8 上个实验的画图
close all;
figure(1);
N_bins = 40;
subplot(2,1,1);
histogram(PSNR_vec_noiseless,N_bins,'FaceColor',"#77AC30");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [new frame,noiseless,5 class]',mus_noiseless),'FontSize',12);

subplot(2,1,2);
histogram(PSNR_vec,N_bins,'FaceColor',"#0072BD");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [new frame,2%% noise,5 class]',mus),'FontSize',12);


%% 2024-5-8 10类数据集重建：无噪与带噪
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
choice = 1; %4为带噪
nls = 0.02;
map_level = nls;
read_level = nls;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0507-10class\val-100k-rgb\';
target_root = '.\ImageNet\0507-10class\new-frame\recons-val-100k-rgb-0508-noiseless\';
% target_root = ['.\ImageNet\0321-5class\new-frame\recons-val-100k-rgb-0508-noise-',num2str(nls),'\'];
% source_root = '.\ImageNet\selected\original\';
% target_root = ['.\ImageNet\selected\recons-new-frame-0508-noise-',num2str(nls),'\'];

class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 10;                         
save_flag = 1;

[PSNRs_noiseless] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);


[mus_noiseless,sigmas_noiseless,PSNR_vec_noiseless] = PSNR_evaluate(PSNRs_noiseless);

choice = 4; %4为带噪
target_root = ['.\ImageNet\0507-10class\new-frame\recons-val-100k-rgb-0508-noise-',num2str(nls),'\'];
[PSNRs] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);


[mus,sigmas,PSNR_vec] = PSNR_evaluate(PSNRs);

save('.\save-mat\psnr_0508_10class_new_frame.mat','mus_noiseless','sigmas_noiseless','PSNR_vec_noiseless','mus','sigmas','PSNR_vec');



%% 2024-5-8 上个实验的画图
close all;
figure(1);
N_bins = 40;
subplot(2,1,1);
histogram(PSNR_vec_noiseless,N_bins,'FaceColor',"#77AC30");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [new frame,noiseless,10 class]',mus_noiseless),'FontSize',12);

subplot(2,1,2);
histogram(PSNR_vec,N_bins,'FaceColor',"#0072BD");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [new frame,2%% noise,10 class]',mus),'FontSize',12);


%% 2024-5-9 5类-扫描noise level
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);
source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
choice = 4;
nls = [0.01,0.03:0.01:0.10];
L = length(nls);
mus_list = zeros(1,L);
sigmas_list = zeros(1,L);
PSNR_vec_list = cell(1,L);
for i = 1:1:L
    nl_now = nls(i);
    map_level = nl_now;
    read_level = nl_now;
    target_root = ['.\ImageNet\0321-5class\new-frame\recons-val-100k-rgb-0508-noise-',num2str(nl_now),'\'];
    [PSNRs] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);
    [mus_list(i),sigmas_list(i),PSNR_vec_list{i}] = PSNR_evaluate(PSNRs);
end
save('.\save-mat\psnr_0508_5class_new_frame.mat','mus_list','sigmas_list','PSNR_vec_list');

%% 2024-5-9 上个实验的画图
close all;
nls_now = 0:0.01:0.10;
figure(1);
PSNR_mu = xlsread('noise_level.xlsx','2','D4:N4');
plot(nls_now,PSNR_mu,'-*','linewidth',1.5);
grid;
xlim([-0.01 nls_now(end)+0.01]); 
xticks(nls_now);
ylim([10 32]);
xlabel('noise level','Fontsize',12);
ylabel('avg-PSNR','Fontsize',12);
title('5 class dataset PSNR(avg)-noise level curve','FontSize',15);

figure(2);
nets = {'ResNet18','ResNet34','ResNet50','ResNet101','ResNet152'};
results = xlsread('noise_level.xlsx','2','D7:N11');
hold on;
for i=1:1:5
    plot(nls_now,results(i,:),'-*','linewidth',1.2,'DisplayName', sprintf(nets{i}));
end
hold off;
grid;
legend show;
xlim([-0.01 nls_now(end)+0.01]); 
xticks(nls_now);
xlabel('noise level','Fontsize',12);
ylabel('classification accuracy','Fontsize',12);
title('5 class dataset accuracy-noise level curve','FontSize',15);


%% 2024-5-10 10类-扫描noise level
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);
source_root = '.\ImageNet\0507-10class\val-100k-rgb\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 10;                         
save_flag = 1;
choice = 4;
nls = [0.01,0.03:0.01:0.10];
L = length(nls);
mus_list = zeros(1,L);
sigmas_list = zeros(1,L);
PSNR_vec_list = cell(1,L);

% 计划分多次重建，全部完成要大概20小时
begin_index = 7;
end_index = 9;

for i = begin_index:1:end_index
    nl_now = nls(i);
    map_level = nl_now;
    read_level = nl_now;
    target_root = ['.\ImageNet\0507-10class\new-frame\recons-val-100k-rgb-0509-noise-',num2str(nl_now),'\'];
    [PSNRs] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                        handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,Gauss_noise_map_NN,Gauss_noise_read_NN,...
                        source_root,target_root,class_name,begin_class,end_class,save_flag);
    [mus_list(i),sigmas_list(i),PSNR_vec_list{i}] = PSNR_evaluate(PSNRs);
    
    % 9个值分为_1:0.01,0.03 _2:0.04-0.07 _3:0.08-0.10
    save('.\save-mat\psnr_0508_10class_new_frame_3.mat','mus_list','sigmas_list','PSNR_vec_list');
    
end

%% 2024-5-10 上个实验的画图

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

%% meas-matrix and trans_matrix generate 2024-4-21
function Phi = meas_matrix_generate(M,N,d,flag,orth_flag)
    % Gauss
    if flag == 1
        rng(11637);
        Phi = randn(M,N) * sqrt(1/M);
    % 0,1
    elseif flag == 2
        rng(11637);
        Phi = zeros(M,N);
        for i = 1:1:N
            col_idx = randperm(M);
            Phi(col_idx(1:d),i) = 1;
        end
    % Bernouli
    elseif flag == 3
        Phi = randi([0,1],M,N);
        %If your MATLAB version is too low,please use randint instead
        Phi(Phi==0) = -1;
        Phi = Phi/sqrt(M);
    else
        error('Please type in 1,2,or 3');
    end
    
    % orth
    if orth_flag == 1
        Phi = orth(Phi')';
    end
end

function Psi = trans_matrix_generate(N,flag)
    %dct
    if flag == 0
        Psi = eye(N);
    elseif flag == 1
        Psi = dctmtx(N);
    else
        error('Please type in 0 or 1');
    end
end


%% 展一个生成随机数的函数，输入M,N，生成两种矩阵库 2024-5-4 
function [Gauss_noise_map,Gauss_noise_read] = iters_generate(M,N,seed)
    Gauss_noise_read = zeros(M,N,100);
    rng(seed);
    Gauss_noise_map = randn(M,N);
    rng('shuffle');
    for i = 1:1:100
        Gauss_noise_read(:,:,i) = randn(M,N);
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

%% 1D and 2D空域压缩感知核心代码 2024-4-24
% 1D-AMP for sparse vector (new_version)
function [x,h] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,noise_info,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN)
    %x: original signal
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
    
    %h: N*1 sparse signal reconstruction result
    %b: M*1 residual(during iteration)
    %MSE/NMSE: MSE/NMSE change in the iteration process
     
    map_noise_flag = noise_info(1);
    map_noise_level = noise_info(2); % typical, or 1% (optimized)
    read_noise_flag = noise_info(3);
    read_noise_level = noise_info(4); % typical, or 1% (optimized)
    meas_noise_flag = noise_info(5);
    recons_noise_flag = noise_info(6);
    trans_noise_flag = noise_info(7);
    
    [M,N] = size(Phi);
    MSE = [];
    NMSE = [];
    
    % compress and sparse ground_truth
    % Phi如何加噪声？
    if meas_noise_flag == 1
        % rng(11637);
        if map_noise_flag
            %noise = max(abs(Phi(:)))*map_noise_level*randn(M,N);
            noise = max(abs(Phi(:)))*map_noise_level*noise_map_MN;
            Phi_map_noise = Phi + noise;
        else
            Phi_map_noise = Phi;
        end
        % rng('shuffle');
        if read_noise_flag
            %noise = max(abs(Phi(:)))*read_noise_level*randn(M,N);
            noise = max(abs(Phi(:)))*read_noise_level*noise_read_MN(:,:,randi(100));
            Phi_noise = Phi_map_noise + noise;
        else
            Phi_noise = Phi_map_noise;
        end
    else
        Phi_noise = Phi;
    end
    % y = Phi_noise * x_now;
    [~,y] = fir_self(Phi_noise * x_now, fir0);
    
    h_truth = Psi * x_now; % h_truth not exist in real dataflow
    A = Phi * Psi';
 
    h0 = zeros(N,1);
    b0 = y;
    
    % map deviation
    if recons_noise_flag == 1
        % rng(11637);
        if map_noise_flag
            %noise = max(abs(A(:)))*map_noise_level*randn(M,N);
            noise = max(abs(A(:)))*map_noise_level*noise_map_MN;
            A_map_noise = A + noise;
        else
            A_map_noise = A;
        end
        % rng('shuffle');
    else
        A_map_noise = A;
    end
    
    %iteration
    % h0,b0→h,b
    for t = 1:1:num
        
        lambda0 = alpha*norm(b0,2) / sqrt(M);
        
        c0 = sum(h0(:)~=0) / M;   %AMP append element  
        
        % read deviation MVM1 and fir1
        if recons_noise_flag == 1
            if read_noise_flag
                % noise = max(abs(A(:)))*read_noise_level*randn(M,N);
                noise = max(abs(A(:)))*read_noise_level*noise_read_MN(:,:,t);
                A_noise = A_map_noise + noise;
            else
                A_noise = A_map_noise;
            end
        else
            A_noise = A;
        end

        [~,r0] = fir_self(A_noise' * b0, fir1);
        
        h = amp_eta_t(r0 + h0, lambda0);
        
        % h_fir1
        h = h.*h_fir1;
        
        % read deviation MVM2 and fir2
%         if read_noise_flag
%             noise = max(abs(A(:)))*read_noise_level*randn(M,N);
%             A_noise = A_map_noise + noise;
%         else
%             A_noise = A_map_noise;
%         end
        
        [~,s] = fir_self(A_noise * h, fir2);
        
        b = y - s + b0 * c0;
        
        %recording MSE and NMSE change
        MSE_now = (norm(h - h_truth)^2)/N;
        NMSE_now = (norm(h - h_truth)^2)/(norm(h_truth)^2);
        MSE = [MSE,MSE_now];
        NMSE = [NMSE,NMSE_now];
        epsilon_now = norm((h - h0),2) / (norm(h,2)+1e-8);
        
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
        h0 = h;
        b0 = b;
    end
    
    % x_fir2
    h = h.*h_fir2;
    
    % sparse domain to original domain
    % 这里是否也需要一个滤波？
    if trans_noise_flag == 1
        % rng(11637);
        if map_noise_flag
            % noise = max(abs(Psi(:)))*map_noise_level*randn(N,N);
            noise = max(abs(Psi(:)))*map_noise_level*noise_map_NN;
            Psi_map_noise = Psi + noise;
        else
            Psi_map_noise = Psi;
        end
        % rng('shuffle');
        if read_noise_flag
            % noise = max(abs(Psi(:)))*read_noise_level*randn(N,N);
            noise = max(abs(Psi(:)))*read_noise_level*noise_read_NN(:,:,randi(100));
            Psi_noise = Psi_map_noise + noise;
        else
            Psi_noise = Psi_map_noise;
        end
    else
        Psi_noise = Psi;
    end
    % y = Phi_noise * x_now;
    [~,x] = fir_self(Psi_noise' * h, fir3);
    % x = Psi' * h;
end

% 2D-AMP for single channel picture compression and recovery
function [PSNR,hat_x] = amp_2d(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN)
    % noise
    noise_info = [0,map_level,0,read_level,1,1,1;
                  1,map_level,0,read_level,1,1,1;
                  0,map_level,1,read_level,1,1,1;
                  1,map_level,1,read_level,1,1,1];
    % choice:only one can be chosen,0001~1111->1,2,3,4
    noise_info_choice = noise_info(choice,:);

    [~,N] = size(Phi);
    hat_x = zeros(N,N);
    x = pic2double(x,N);
    % column wise recons
    if handle_2d == 0
        for i = 1:1:N
            x_now = x(:,i);
            [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,noise_info_choice,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN);
            % hat_x_now = Psi' * hat_h;
            hat_x(:,i) = hat_x_now;
        end  
    elseif handle_2d == 1
        ps = 16;
        num_row = round(N / ps);
        num_col = round(N / ps);
        for i = 1:1:num_row
            for j = 1:1:num_col
                patch_to_handle = x((i-1)*ps+1:i*ps,(j-1)*ps+1:j*ps);
                x_now = reshape(patch_to_handle,[N,1]);
                [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN);
                hat_patch = reshape(hat_x_now,[ps,ps]);
                hat_x((i-1)*ps+1:i*ps,(j-1)*ps+1:j*ps) = hat_patch;
            end
        end
    else
        error('This handle has not been developed,please type in 0');
    end
    PSNR = psnr(x,hat_x);
end

% 2D-AMP for 3-channel RGB picture compression and recovery
function [PSNR,hat_x_rgb] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN)
    % need to be rgb
    [~,N,color] = size(x);
    if(color~=3)
        error('Picture Must be RGB three channel.')
    end
    % 3 channels reconstruction
    hat_x_rgb = zeros(N,N,3);
    PSNR_now = zeros(1,3);
    for i = 1:1:3
        x_now = x(:,:,i);
        [PSNR_now(i),hat_x_rgb(:,:,i)] = amp_2d(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
end

% 数据集重建函数 2024-5-8
function [PSNRs] = amp_2d_recons(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                                 handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN,...
                                 source_root,target_root,class_name,begin_class,end_class,save_flag)
    
    PSNRs = cell(1,end_class - begin_class + 1);
    % [~,N] = size(Phi); 
    
    % class-wise handle
    for i = begin_class:1:end_class
        tic;
        % create target folder
        class = [(class_name{i}),'\'];
        if ~(isfolder([target_root,class]))
            disp('target folder doesnt exist,create it.');
            mkdir([target_root,class]);
            disp('successfully create.')
        end
        
        % read picture set
        images = dir(fullfile(source_root,class,'*.png'));
        images_num = length(images);
        i_real = i - begin_class + 1;
        % PSNRs{i_real} = zeros(1,images_num);
        PSNRs{i_real} = [];
        
        % pic-wise handle
        for j = 1:1:images_num
            % read
            pic_name = images(j).name;   
            x = imread([source_root,class,pic_name]);
            x = im2double(x);
            [~,~,color] = size(x);            
            
            % 1 or 3 channel
            if color == 1
                % reconstruction
                [PSNR_now, hat_x_now] =  amp_2d(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                                             handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN);
                % if then save
                if save_flag == 1
                    imwrite(hat_x_now,[target_root,class,pic_name]);
                end
            elseif color == 3
                % reconstruction
                [PSNR_now, hat_x_now] =  amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,fir3,h_fir1,h_fir2,map_level,read_level,choice,...
                                             handle_2d,stop_earlier,show_detail,noise_map_MN,noise_read_MN,noise_map_NN,noise_read_NN);
                % if then save
                if save_flag == 1
                    imwrite(hat_x_now,[target_root,class,pic_name]);
                end
            else
                error('Channels must be 1 or 3');
            end
            
            % print and results
            PSNRs{i_real} = [PSNRs{i_real},PSNR_now];
            fprintf('class %s test recons-progress : %d in %d ,PSNR = %.2fdB,class: %d in %d\n',class_name{i},j,images_num,PSNR_now,i-begin_class,end_class-begin_class);
        end
        toc;
    end
end

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

