%% 2024-4-24 use this to implement trans-domain measurement CS 2024-4-24
%% 2024-4-25 直接截断 
clear all;
close all;
load ('.\save-mat\params_0424.mat');
% save('.\save-mat\params_0424.mat');
trans_type = 0; %%
meas_noise_flag = 0; %%
handle_2d = 0;
M = round(N * cr);
Phi = meas_matrix_generate(M,N,16,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

%Nts = zeros(5,1);
PSNR_origin = zeros(5,1);
PSNR_nl = zeros(5,1);
hat_x_cell = {};
tic;
for i = 1:1:5
    x = im2double(imread(['.\pre-pic\',num2str(i),'.png']));
    x = imresize(x,[N,N]);
    h = dct2(x);
    % [~,~,color] = size(h);
    choice = 1;
    h_fir2 = ones(N,1);
    [~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
    hat_x = idct2(hat_h);
    PSNR_origin(i) = psnr(x,hat_x);
    
    choice = 4;
    map_level = 0.02;
    read_level = 0.02;
    % [~,~,color] = size(h);
    [~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
    hat_x = idct2(hat_h);
    PSNR_nl(i) = psnr(x,hat_x);
    
    limit = 16:8:256;
    %limit = 96:8:144;
    PSNR_test = zeros(5,length(limit));
    for j = 1:1:length(limit)
        h_fir2 = [ones(limit(j),1);zeros(N-limit(j),1)];
        [~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
        hat_x = idct2(hat_h);
        PSNR_test(i,j) = psnr(x,hat_x);
        hat_x_cell{j} = hat_x;
    end
    figure(i);
    plot(limit,PSNR_test(i,:),'*-','LineWidth',1.5);
    grid;
    hold on;
    plot(limit,PSNR_nl(i)*ones(1,length(limit)),'--','LineWidth',1.5);
    plot(limit,PSNR_origin(i)*ones(1,length(limit)),'--','LineWidth',1.5);
    xlabel('simple truncate number N_t','FontSize',12);
    ylabel('PSNR/dB','FontSize',12);
    legend('sim-truncated','sim-no truncated','software','FontSize',10);
    xlim([8 264]);xticks(16:16:256);
    title('2D AMP reconstruction PSNR-N_t curve','FontSize',12);
    [val,find] = max(PSNR_test(i,:));
    fprintf('pic %d soft PSNR=%.2f,sim PSNR=%.2f,best PSNR=%.2f at N_t=%d\n',i,PSNR_origin(i),PSNR_nl(i),val,limit(find));
end
toc;


%% 2024-4-25 固定阈值截断 
% clear all;
% PSNR_nl= [24.14;23.74;22.74;27.29;26.69];
% PSNR_origin = [29.85;32.52;24.65;40.63;46.85];
% save('.\save-mat\pic1_5-sim1.mat');
clear all;
close all;
load ('.\save-mat\params_0424.mat');
load('.\save-mat\pic1_5-sim1.mat');
% save('.\save-mat\params_0424.mat');
trans_type = 0; %%
meas_noise_flag = 0; %%
handle_2d = 0;
M = round(N * cr);
Phi = meas_matrix_generate(M,N,16,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);
%Nts = zeros(5,1);
limit = 0.005:0.005:0.1;
PSNR_test = zeros(5,length(limit));

for i = 1:1:1
    tic;
    x = im2double(imread(['.\pre-pic\',num2str(i),'.png']));
    x = imresize(x,[N,N]);
    h = dct2(x);
    % [~,~,color] = size(h);
    choice = 4;
    map_level = 0.02;
    read_level = 0.02;
 
    for j = 1:1:length(limit)
        trunc = [1,limit(i)];
        [~,hat_h] = amp_2d_t(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,trunc,choice,handle_2d,stop_earlier,show_detail);
        hat_x = idct2(hat_h);
        PSNR_test(i,j) = psnr(x,hat_x);
    end
    figure(i);
    plot(limit,PSNR_test(i,:),'*-','LineWidth',1.5);
    grid;
    hold on;
    plot(limit,PSNR_nl(i)*ones(1,length(limit)),'--','LineWidth',1.5);
    plot(limit,PSNR_origin(i)*ones(1,length(limit)),'--','LineWidth',1.5);
    xlabel('truncate threshold t','FontSize',12);
    ylabel('PSNR/dB','FontSize',12);
    legend('sim-truncated','sim-no truncated','software','FontSize',10);
    xlim([limit(1)-0.005 limit(end)+0.005]);xticks(0:0.01:0.1);
    title('2D AMP reconstruction PSNR-t curve','FontSize',12);
    [val,find] = max(PSNR_test(i,:));
    fprintf('pic %d soft PSNR=%.2f,sim PSNR=%.2f,best PSNR=%.2f at t=%.3f\n',i,PSNR_origin(i),PSNR_nl(i),val,limit(find));
    
    toc;
end

    
%% 2024-4-25 其他测量矩阵 一维
clear all;
close all;
load ('.\save-mat\params_0424.mat');
% save('.\save-mat\params_0424.mat');

M = round(N * cr);
K = 48;
x = zeros(N,1);
x(randsample(1:N,K)) = randn(1,K);

trans_type = 0;
meas_noise_flag = 0;
meas_type = 1;
orth_flag = 1;
d = 16;
Phi = meas_matrix_generate(M,N,d,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);
[hat_x,hat_h] = amp_core(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info,meas_noise_flag,stop_earlier,show_detail);
NMSE = (norm(x - hat_x)^2)/(norm(x)^2)

meas_type = 3;
orth_flag = 0;
d = 32;
Phi = meas_matrix_generate(M,N,d,meas_type,orth_flag);
[hat_x_test,hat_h_test] = amp_core(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info,meas_noise_flag,stop_earlier,show_detail);
NMSE_test = (norm(x - hat_x_test)^2)/(norm(x)^2)

%% 2024-4-25 其他测量矩阵 图像
clear all;
close all;
load ('.\save-mat\params_0424.mat');
% save('.\save-mat\params_0424.mat');
trans_type = 0;
meas_noise_flag = 0;
handle_2d = 0;
M = round(N * cr);

meas_type = 2;
orth_flag = 0;
d = 128;
Phi = meas_matrix_generate(M,N,d,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);
PSNR_test_origin= zeros(5,1);
PSNR_test_nl= zeros(5,1);
tic;
for i = 1:1:5
    x = im2double(imread(['.\pre-pic\',num2str(i),'.png']));
    x = imresize(x,[N,N]);
    h = dct2(x);
    
    choice = 1;
    [~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
    hat_x = idct2(hat_h);
    PSNR_test_origin(i) = psnr(x,hat_x);
    
    choice = 4;
    map_level = 0.05;
    read_level = 0.05;
    % [~,~,color] = size(h);
    [~,hat_h_nl] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
    hat_x_nl = idct2(hat_h_nl);
    PSNR_test_nl(i) = psnr(x,hat_x_nl);
    
    fprintf('soft PSNR=%.2f, nl PSNR=%.2f, diff PSNR=%.2f\n',PSNR_test_origin(i),PSNR_test_nl(i),PSNR_test_origin(i)-PSNR_test_nl(i));
end
toc;

%% 2024-4-25 其他测量矩阵 图像-噪声
% 1张图 3种测量矩阵 0:0.01:0.10 共30个测量点（尝试用RGB？)
clear all;
close all;
load ('.\save-mat\params_0424.mat');
% save('.\save-mat\params_0424.mat');
trans_type = 0;
meas_noise_flag = 0;
handle_2d = 0;
M = round(N * cr);
choice = 4;
types = [1,3,4];
orths = [1,0,0];
x = im2double(imread('.\pre-pic\6.png'));
x = imresize(x,[N,N]);
%limit = 0;
limit = 0:0.005:0.10;
PSNRs = zeros(3,length(limit));
for i = 1:1:3
    meas_type = types(i);
    orth_flag = orths(i);
    Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
    Psi = trans_matrix_generate(N,trans_type);
    for j = 1:1:length(limit)
        map_level = limit(j);
        read_level = limit(j);
        tic;
        for k = 1:1:3
            x_now = x(:,:,k);
            h_now = dct2(x_now);
            [~,hat_h_now] = amp_2d(h_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
            hat_x_now = idct2(hat_h_now);
            hat_x(:,:,k) = hat_x_now;
        end
        PSNRs(i,j) = psnr(x,hat_x);
        fprintf('meas type %d in %d, noise level %.3f in %.3f ',i,length(types),limit(j),limit(end));
        toc;
        fprintf('\n');
    end
end

save('.\save-mat\psnr_0424_noise_scan.mat');

figure;
plot(limit,PSNRs(1,:),'*-','LineWidth',1.5);
grid;
hold on;
plot(limit,PSNRs(2,:),'o-','LineWidth',1.5);
plot(limit,PSNRs(3,:),'s-','LineWidth',1.5);
xlabel('noise level','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('Gauss+orth','Bernoulli 1','Bernoulli 2','FontSize',10);
xlim([limit(1)-0.005 limit(end)+0.005]);xticks(0:0.01:0.1);
ylim([12 32]);
title('2D AMP reconstruction PSNR-noise level & meas type curve','FontSize',12);


%% 2024-5-3 保存随机数？
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
% seed = 11637;
% tic;
% [Gauss_noise_map,Gauss_noise_read] = iters_generate(M,N,seed);
% toc;
% 
% fir3 = [4,0.7000,1,0];
% save('.\save-mat\params_0504.mat');
meas_noise_flag = 0; %%
trans_type = 0; %%
meas_type = 1; %%
orth_flag = 1; %%
choice = 1;
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

% gray
tic;
x = im2double(imread('.\pre-pic\1.png'));
x = imresize(x,[N,N]);
h = dct2(x);
[~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,...
                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read);
hat_x = idct2(hat_h);
PSNR = psnr(x,hat_x)
toc;

% rgb
tic;
x = im2double(imread('.\pre-pic\6.png'));
x = imresize(x,[N,N]);
hat_x = zeros(N,N,3);
for i = 1:1:3
    h_now = dct2(x(:,:,i));
    [~,hat_h_now] = amp_2d(h_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,...
                        choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read);
    hat_x(:,:,i) = idct2(hat_h_now);
end
PSNR_rgb = psnr(x,hat_x)
toc;


%% 2024-5-3 直接截断-端到端-有截断
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 1; %% 1-高斯
orth_flag = 1; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-single-trunc\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 16:16:256;
[PSNRs,Nts] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);

% save('.\save-mat\psnr_0504_test.mat','PSNRs','Nts')

%% 2024-5-5 直接截断-端到端-无截断
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 1; %% 1-高斯
orth_flag = 1; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-single-truncless\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 0;
%limit = 16:8:256;
limit = 256;
[PSNRs_truncless,Nts_truncless] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);

save('.\save-mat\psnr_0505_test.mat','PSNRs_truncless','Nts_truncless')

%% 2024-5-5 直接截断-端到端-无截断-软件
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 1; %% 1-高斯
orth_flag = 1; %% 1-有正交归一
choice = 1; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-single-truncless\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 0;
%limit = 16:8:256;
limit = 256;
[PSNRs_soft,Nts_soft] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);

save('.\save-mat\psnr_0505_soft.mat','PSNRs_soft','Nts_soft')

%% 2024-5-5 评估
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;

load('.\save-mat\psnr_0504_test.mat');
load('.\save-mat\psnr_0505_test.mat');
load('.\save-mat\psnr_0505_soft.mat');

[mu_soft,sigma_soft,PSNR_vec_soft] = PSNR_evaluate(PSNRs_soft);
[mu_truncless,sigma_truncless,PSNR_vec_truncless] = PSNR_evaluate(PSNRs_truncless);
[mu_trunc,~,PSNR_vec_trunc] = PSNR_evaluate(PSNRs);
[mu_Nt,sigma_Nt,vec_Nt] = PSNR_evaluate(Nts);


%% 2024-5-5 画图
close all;

figure(1);
plot(PSNR_vec_soft,'-*','LineWidth',1.0);
hold on;
plot(PSNR_vec_truncless,'-s','LineWidth',1.0);
plot(PSNR_vec_trunc,'-o','LineWidth',1.0);
grid;
xlabel('Number','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
xlim([-1 251]); xticks(0:50:250);
legend('Noiseless','2% noise','2% noise with trunc');
title('PSNR-pic number curve','FontSize',15)

figure(2);
N_bins = 40;
subplot(3,1,1);
histogram(PSNR_vec_soft,N_bins,'FaceColor',"#77AC30");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [Noiseless]',mu_soft),'FontSize',12);

subplot(3,1,2);
histogram(PSNR_vec_truncless,N_bins,'FaceColor',"#0072BD");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [2%% noise]',mu_truncless),'FontSize',12);

subplot(3,1,3);
histogram(PSNR_vec_trunc,N_bins,'FaceColor',"#EDB120");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [2%% noise with trunc]',mu_trunc),'FontSize',12);


figure(3);
subplot(2,1,1);
histogram(vec_Nt,'BinEdges',8:16:264,'FaceColor',"#EDB120");
xlim([8,264]);
xticks(16:16:256);
xlabel('Nt','FontSize',12);
ylabel('frequency','FontSize',12);
title(sprintf('Nt distribution with avg-Nt = %d',round(mu_Nt)),'FontSize',12);

subplot(2,1,2);
PSNR_prog = (PSNR_vec_trunc - PSNR_vec_truncless) ./ (PSNR_vec_soft - PSNR_vec_truncless);
plot(vec_Nt,PSNR_prog,'*','LineWidth',1.0);
xlabel('Nt','FontSize',12);
ylabel('PSNR improvement ratio','FontSize',10);
xlim([8,264]);
xticks(16:16:256);
title(sprintf('PSNR improvement ratio distribution with avg-ratio = %.2f%%',mean(PSNR_prog)*100),'FontSize',12);


%% 2024-5-6 Nt=128
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 1; %% 1-高斯
orth_flag = 1; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-trunc-128\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 128;
[PSNRs_Nt_128,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);

save('.\save-mat\psnr_0505_Nt_128.mat','PSNRs_Nt_128','mu_Nt_128','sigma_Nt_128','PSNR_vec_Nt_128');
[mu_Nt_128,sigma_Nt_128,PSNR_vec_Nt_128] = PSNR_evaluate(PSNRs_Nt_128);


%% 2024-5-6 Nt=176
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 1; %% 1-高斯
orth_flag = 1; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-trunc-176\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 176;
[PSNRs_Nt_176,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);


[mu_Nt_176,sigma_Nt_176,PSNR_vec_Nt_176] = PSNR_evaluate(PSNRs_Nt_176);

save('.\save-mat\psnr_0505_Nt_176.mat','PSNRs_Nt_176','mu_Nt_176','sigma_Nt_176','PSNR_vec_Nt_176');


%% 2024-5-6 伯努利矩阵测试
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 3; %% 1-高斯
orth_flag = 0; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-bernouli\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 256;
[PSNRs_ber,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);


[mu_ber,sigma_ber,PSNR_vec_ber] = PSNR_evaluate(PSNRs_ber);

save('.\save-mat\psnr_0505_ber.mat','PSNRs_ber','mu_ber','sigma_ber','PSNR_vec_ber');

%% 2024-5-6 伯努利矩阵测试-无噪
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 3; %% 1-高斯
orth_flag = 0; %% 1-有正交归一
choice = 1; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-bernouli-noiseless\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 256;
[PSNRs_ber_nl,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);


[mu_ber_nl,sigma_ber_nl,PSNR_vec_ber_nl] = PSNR_evaluate(PSNRs_ber_nl);

save('.\save-mat\psnr_0506_ber_nl.mat','PSNRs_ber_nl','mu_ber_nl','sigma_ber_nl','PSNR_vec_ber_nl');

%% 2024-5-6 伯努利2矩阵测试-无噪
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 4; %% 1-高斯
orth_flag = 0; %% 1-有正交归一
choice = 1; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-bernouli2-noiseless\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 256;
[PSNRs_ber2_nl,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);


[mu_ber2_nl,sigma_ber2_nl,PSNR_vec_ber2_nl] = PSNR_evaluate(PSNRs_ber2_nl);

save('.\save-mat\psnr_0506_ber2_nl.mat','PSNRs_ber2_nl','mu_ber2_nl','sigma_ber2_nl','PSNR_vec_ber2_nl');

%% 2024-5-6 伯努利2矩阵测试-有噪
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;
meas_noise_flag = 0; %%
trans_type = 0; %% 0-dct域采样
meas_type = 4; %% 1-高斯
orth_flag = 0; %% 1-有正交归一
choice = 4; %% 1-无噪 4-有噪
Phi = meas_matrix_generate(M,N,32,meas_type,orth_flag);
Psi = trans_matrix_generate(N,trans_type);

source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
%source_root = '.\ImageNet\0321-5class\selected\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb-0504-bernouli2\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 1;
%limit = 16:8:256;
limit = 256;
[PSNRs_ber2,~] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,Gauss_noise_map,Gauss_noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit);


[mu_ber2,sigma_ber2,PSNR_vec_ber2] = PSNR_evaluate(PSNRs_ber2);

save('.\save-mat\psnr_0506_ber2.mat','PSNRs_ber2','mu_ber2','sigma_ber2','PSNR_vec_ber2');

%% 2024-5-6 评估与画图
clear all;
close all;
clc;
load('.\save-mat\params_0504.mat');
M = cr * N;

load('.\save-mat\psnr_0504_test.mat');
load('.\save-mat\psnr_0505_Nt_128.mat');
load('.\save-mat\psnr_0505_Nt_176.mat');

[mu_trunc,sigma_trunc,PSNR_vec_trunc] = PSNR_evaluate(PSNRs);

figure(1);
N_bins = 40;
subplot(3,1,3);
histogram(PSNR_vec_Nt_176,N_bins,'FaceColor',"#EDB120");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [2%% noise with Nt=176-trunc]',mu_Nt_176),'FontSize',12);

subplot(3,1,2);
histogram(PSNR_vec_Nt_128,N_bins,'FaceColor',"#0072BD");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [2%% noise with Nt=128-trunc]',mu_Nt_128),'FontSize',12);

subplot(3,1,1);
histogram(PSNR_vec_trunc,N_bins,'FaceColor',"#77AC30");
xlim([10,50]);
xlabel('PSNR/dB','FontSize',12);ylabel('frequency','FontSize',12);
title(sprintf('PSNR distribution with avg-PSNR = %.2fdB [2%% noise with optimal Nt-trunc]',mu_trunc),'FontSize',12);

%% 2024-5-8 条件数和MRM计算
clear all;
close all;
clc;
N = 256;
M = 192;
Ls = [32,32,32,32,0:1:6];
meas_types = [1,1,3,4,5*ones(1,7)];
orth_flags = [1,0,0,0,zeros(1,7)];
Psi = trans_matrix_generate(N,1);
L = length(Ls);
cond_Phi = zeros(1,L);
cond_A = zeros(1,L);
tic;
for i = 1:1:L
    Phi = meas_matrix_generate(M,N,Ls(i),meas_types(i),orth_flags(i));
    cond_Phi(i) = cond(Phi);
    % sum(sum(abs(Phi.*Phi)))
    A = Phi * Psi;
    cond_A(i) = cond(A);
    sum(sum(abs(A.*A)))
end 
toc;
cond_Phi
cond_A

%% 

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
            Phi(col_idx(1:d),i) = 1/sqrt(d);
        end
    % Bernouli
    elseif flag == 3
        rng(11637);
        Phi = randi([0,1],M,N);
        %If your MATLAB version is too low,please use randint instead
        Phi(Phi==0) = -1;
        Phi = Phi/sqrt(M);
    elseif flag == 4
        rng(11637);
        Phi = randi([-1,4],M,N);%If your MATLAB version is too low,please use randint instead
        Phi(Phi==2) = 0;%P=1/6
        Phi(Phi==3) = 0;%P=1/6
        Phi(Phi==4) = 0;%P=1/6
        Phi = Phi*sqrt(3/M);
    elseif flag == 5
        % d在此处=L
        % 
        % 生成p=1/(d+2)的伯努利矩阵
        rng(11637);
        Phi = randi([-1,d],M,N);
        if d < 1
            Phi(Phi==0) = 1;
        elseif d == 1
            %Phi = Phi;
        else
            for i = 2:1:d
                Phi(Phi==i) = 0;
            end
        end
        Phi = Phi * sqrt((d+2) / 2 / M);
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

%% amp-1d and 2d compression and reconstruction with original domain measurement 2024-4-24
function [] = buff1()
% % 1D-AMP for sparse vector (new_version)
% function [x,h] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info,meas_noise_flag,stop_earlier,show_detail)
%     %x: original signal
%     %y: M*1 signal after measurement
%     %A: Phi * Psi
%     %num: max iteration num
%     %epsilon: min iteration step
%     %alpha: usually alpha=1 for this fucntion
%     %noise_info: [map_noise_flag,map_level,read_noise_flag,read_level]...
%     %...flag==0 for noiseless
%     %theta: ground truth for sparse vector to be reconstructed]
%     %stop_earlier: no need for run all num iteration...
%     %...if less than epsilon, break.
%     
%     %h: N*1 sparse signal reconstruction result
%     %b: M*1 residual(during iteration)
%     %MSE/NMSE: MSE/NMSE change in the iteration process
%      
%     map_noise_flag = noise_info(1);
%     map_noise_level = noise_info(2); % typical, or 1% (optimized)
%     read_noise_flag = noise_info(3);
%     read_noise_level = noise_info(4); % typical, or 1% (optimized)
%     
%     [M,N] = size(Phi);
%     MSE = [];
%     NMSE = [];
%     
%     % compress and sparse ground_truth
%     % Phi如何加噪声？
%     if meas_noise_flag == 1
%         rng(11637);
%         if map_noise_flag
%             noise = max(abs(Phi(:)))*map_noise_level*randn(M,N);
%             Phi_map_noise = Phi + noise;
%         else
%             Phi_map_noise = Phi;
%         end
%         rng('shuffle');
%         if read_noise_flag
%             noise = max(abs(Phi(:)))*read_noise_level*randn(M,N);
%             Phi_noise = Phi_map_noise + noise;
%         else
%             Phi_noise = Phi_map_noise;
%         end
%     else
%         Phi_noise = Phi;
%     end
%     % y = Phi_noise * x_now;
%     [~,y] = fir_self(Phi_noise * x_now, fir0);
%     
%     h_truth = Psi * x_now; % h_truth not exist in real dataflow
%     A = Phi * Psi';
%  
%     h0 = zeros(N,1);
%     b0 = y;
%     
%     % map deviation
%     rng(11637);
%     if map_noise_flag
%         noise = max(abs(A(:)))*map_noise_level*randn(M,N);
%         A_map_noise = A + noise;
%     else
%         A_map_noise = A;
%     end
%     rng('shuffle');
%     
%     %iteration
%     % h0,b0→h,b
%     for t = 1:1:num
%         
%         lambda0 = alpha*norm(b0,2) / sqrt(M);
%         
%         c0 = sum(h0(:)~=0) / M;   %AMP append element  
%         
%         % read deviation MVM1 and fir1
%         if read_noise_flag
%             noise = max(abs(A(:)))*read_noise_level*randn(M,N);
%             A_noise = A_map_noise + noise;
%         else
%             A_noise = A_map_noise;
%         end
% 
%         [~,r0] = fir_self(A_noise' * b0, fir1);
%         
%         h = amp_eta_t(r0 + h0, lambda0);
%         
%         % h_fir1
%         h = h.*h_fir1;
%         
%         % read deviation MVM2 and fir2
% %         if read_noise_flag
% %             noise = max(abs(A(:)))*read_noise_level*randn(M,N);
% %             A_noise = A_map_noise + noise;
% %         else
% %             A_noise = A_map_noise;
% %         end
%         
%         [~,s] = fir_self(A_noise * h, fir2);
%         
%         b = y - s + b0 * c0;
%         
%         %recording MSE and NMSE change
%         MSE_now = (norm(h - h_truth)^2)/N;
%         NMSE_now = (norm(h - h_truth)^2)/(norm(h_truth)^2);
%         MSE = [MSE,MSE_now];
%         NMSE = [NMSE,NMSE_now];
%         epsilon_now = norm((h - h0),2) / (norm(h,2)+1e-8);
%         
%         % show detail
%         if show_detail == 1
%             fprintf('进行第%d次迭代,epsilon = %4f \n',t,epsilon_now);
%         end
%         
%         % stop earlier
%         if(epsilon_now < epsilon)
%             if(stop_earlier==1)       
%                 if show_detail == 1
%                     fprintf('在%d次迭代后收敛,结束循环\n',t); 
%                 end
%                 break;
%             end
%         end      
%         h0 = h;
%         b0 = b;
%     end
%     
%     % x_fir2
%     h = h.*h_fir2;
%     
%     % sparse domain to original domain
%     x = Psi' * h;
% end
% 
% % 2D-AMP for single channel picture compression and recovery
% function [PSNR,hat_x] = amp_2d(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail)
%     % noise
%     noise_info = [0,map_level,0,read_level;
%                   1,map_level,0,read_level;
%                   0,map_level,1,read_level;
%                   1,map_level,1,read_level];
%     % choice:only one can be chosen,0001~1111->1,2,3,4
%     noise_info_choice = noise_info(choice,:);
% 
%     [~,N] = size(Phi);
%     hat_x = zeros(N,N);
%     x = pic2double(x,N);
%     % column wise recons
%     if handle_2d == 0
%         for i = 1:1:N
%             x_now = x(:,i);
%             [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,stop_earlier,show_detail);
%             % hat_x_now = Psi' * hat_h;
%             hat_x(:,i) = hat_x_now;
%         end  
%     elseif handle_2d == 1
%         ps = 16;
%         num_row = round(N / ps);
%         num_col = round(N / ps);
%         for i = 1:1:num_row
%             for j = 1:1:num_col
%                 patch_to_handle = x((i-1)*ps+1:i*ps,(j-1)*ps+1:j*ps);
%                 x_now = reshape(patch_to_handle,[N,1]);
%                 [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,stop_earlier,show_detail);
%                 hat_patch = reshape(hat_x_now,[ps,ps]);
%                 hat_x((i-1)*ps+1:i*ps,(j-1)*ps+1:j*ps) = hat_patch;
%             end
%         end
%     else
%         error('This handle has not been developed,please type in 0');
%     end
%     PSNR = psnr(x,hat_x);
% end
% 
% % 2D-AMP for 3-channel RGB picture compression and recovery
% function [PSNR,hat_x_rgb] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail)
%     % need to be rgb
%     [~,N,color] = size(x);
%     if(color~=3)
%         error('Picture Must be RGB three channel.')
%     end
%     % 3 channels reconstruction
%     hat_x_rgb = zeros(N,N,3);
%     PSNR_now = zeros(1,3);
%     for i = 1:1:3
%         x_now = x(:,:,i);
%         [PSNR_now(i),hat_x_rgb(:,:,i)] = amp_2d(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail);
%     end
%     % PSNR for rgb: single average
%     PSNR = mean(PSNR_now,2);
% end

end

%% amp with trunc 2024-4-25
% 1D-AMP for sparse vector (new_version)
function [x,h] = amp_core_t(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info,meas_noise_flag,trunc,stop_earlier,show_detail)
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
    
    [M,N] = size(Phi);
    MSE = [];
    NMSE = [];
    
    % compress and sparse ground_truth
    % Phi如何加噪声？
    if meas_noise_flag == 1
        rng(11637);
        if map_noise_flag
            noise = max(abs(Phi(:)))*map_noise_level*randn(M,N);
            Phi_map_noise = Phi + noise;
        else
            Phi_map_noise = Phi;
        end
        rng('shuffle');
        if read_noise_flag
            noise = max(abs(Phi(:)))*read_noise_level*randn(M,N);
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
    rng(11637);
    if map_noise_flag
        noise = max(abs(A(:)))*map_noise_level*randn(M,N);
        A_map_noise = A + noise;
    else
        A_map_noise = A;
    end
    rng('shuffle');
    
    %iteration
    % h0,b0→h,b
    for t = 1:1:num
        
        lambda0 = alpha*norm(b0,2) / sqrt(M);
        
        c0 = sum(h0(:)~=0) / M;   %AMP append element  
        
        % read deviation MVM1 and fir1
        if read_noise_flag
            noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
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
    
    % 截断时候加上这段代码
    trunc_flag = trunc(1);
    trunc_thresh = trunc(2);
    if trunc_flag == 1
        h(abs(h)<trunc_thresh) = 0;
    end
    % sparse domain to original domain
    x = Psi' * h;
end

% 2D-AMP for single channel picture compression and recovery
function [PSNR,hat_x] = amp_2d_t(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,trunc,choice,handle_2d,stop_earlier,show_detail)
    % noise
    noise_info = [0,map_level,0,read_level;
                  1,map_level,0,read_level;
                  0,map_level,1,read_level;
                  1,map_level,1,read_level];
    % choice:only one can be chosen,0001~1111->1,2,3,4
    noise_info_choice = noise_info(choice,:);

    [~,N] = size(Phi);
    hat_x = zeros(N,N);
    x = pic2double(x,N);
    % column wise recons
    if handle_2d == 0
        for i = 1:1:N
            x_now = x(:,i);
            [hat_x_now,~] = amp_core_t(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,trunc,stop_earlier,show_detail);
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
                [hat_x_now,~] = amp_core_t(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,trunc,stop_earlier,show_detail);
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
function [PSNR,hat_x_rgb] = amp_2d_rgb_t(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,trunc,choice,handle_2d,stop_earlier,show_detail)
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
        [PSNR_now(i),hat_x_rgb(:,:,i)] = amp_2d_t(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,trunc,choice,handle_2d,stop_earlier,show_detail);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
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

%% 基于矩阵库的优化amp重建代码
% 1D-AMP for sparse vector (new_version)
function [x,h] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info,meas_noise_flag,stop_earlier,show_detail,noise_map,noise_read)
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
    
    [M,N] = size(Phi);
    MSE = [];
    NMSE = [];
    
    % compress and sparse ground_truth
    % Phi如何加噪声？
    if meas_noise_flag == 1
        %rng(11637);
        if map_noise_flag
            %noise = max(abs(Phi(:)))*map_noise_level*randn(M,N);
            noise = max(abs(Phi(:)))*map_noise_level*noise_map;
            Phi_map_noise = Phi + noise;
        else
            Phi_map_noise = Phi;
        end
        %rng('shuffle');
        if read_noise_flag
            %noise = max(abs(Phi(:)))*read_noise_level*randn(M,N);
            noise = max(abs(Phi(:)))*read_noise_level*noise_read(:,:,randi(100));
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
    %rng(11637);
    if map_noise_flag
        %noise = max(abs(A(:)))*map_noise_level*randn(M,N);
        noise = max(abs(A(:)))*map_noise_level*noise_map;
        A_map_noise = A + noise;
    else
        A_map_noise = A;
    end
    %rng('shuffle');
    
    %iteration
    % h0,b0→h,b
    for t = 1:1:num
        
        lambda0 = alpha*norm(b0,2) / sqrt(M);
        
        c0 = sum(h0(:)~=0) / M;   %AMP append element  
        
        % read deviation MVM1 and fir1
        if read_noise_flag
            %noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            noise = max(abs(A(:)))*read_noise_level*noise_read(:,:,t);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
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
    x = Psi' * h;
end

% 2D-AMP for single channel picture compression and recovery
function [PSNR,hat_x] = amp_2d(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read)
    % noise
    noise_info = [0,map_level,0,read_level;
                  1,map_level,0,read_level;
                  0,map_level,1,read_level;
                  1,map_level,1,read_level];
    % choice:only one can be chosen,0001~1111->1,2,3,4
    noise_info_choice = noise_info(choice,:);

    [~,N] = size(Phi);
    hat_x = zeros(N,N);
    x = pic2double(x,N);
    % column wise recons
    if handle_2d == 0
        for i = 1:1:N
            x_now = x(:,i);
            [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,stop_earlier,show_detail,noise_map,noise_read);
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
                [hat_x_now,~] = amp_core(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,noise_info_choice,meas_noise_flag,stop_earlier,show_detail,noise_map,noise_read);
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
function [PSNR,hat_x_rgb] = amp_2d_rgb(x,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read)
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
        [PSNR_now(i),hat_x_rgb(:,:,i)] = amp_2d(x_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
end

function [PSNRs,Nts] = amp_2d_recons_trunc(Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,map_level,read_level,meas_noise_flag,...
                                    choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read,source_root,target_root,class_name,begin_class,end_class,save_flag,limit)

    PSNRs = cell(1,end_class - begin_class + 1);
    Nts = cell(1,end_class - begin_class + 1);
    
    [~,N] = size(Phi);
    
    for i = begin_class:1:end_class
        tic;
        % not change part
        class = [(class_name{i}),'\'];
        % create target folder
        if ~(isfolder([target_root,class]))
            disp('target folder doesnt exist,create it.');
            mkdir([target_root,class]);
            disp('successfully create.')
        end
        images = dir(fullfile(source_root,class,'*.png'));
        images_num = length(images);
        i_real = i - begin_class + 1;
        PSNRs{i_real} = [];
        Nts{i_real} = [];
        %limit = 16:8:256;
        
        for j = 1:1:images_num
            % read
            pic_name = images(j).name;   
            x = imread([source_root,class,pic_name]);
            x = im2double(x);
            [~,~,color] = size(x);
            PSNR_max = 0;
            Nt_max = 0;
            if color == 1
                hat_x_max = zeros(N,N);
                h = dct2(x);
                for k = 1:1:length(limit)
                    h_fir2 = [ones(limit(k),1);zeros(N-limit(k),1)];
                    [~,hat_h] = amp_2d(h,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read);
                    hat_x = idct2(hat_h);
                    PSNR_now = psnr(x,hat_x);
                    if PSNR_now > PSNR_max
                        PSNR_max = PSNR_now;
                        Nt_max = limit(k);
                        hat_x_max = hat_x;
                    end
                end
                if save_flag == 1                   
                    imwrite(hat_x_max,[target_root,class,pic_name]);
                end
            elseif color == 3
                %hat_x = zeros(N,N,3);
                hat_x_max = zeros(N,N,3);
                for k = 1:1:length(limit)
                    h_fir2 = [ones(limit(k),1);zeros(N-limit(k),1)];
                    for m = 1:1:3
                        h_now = dct2(x(:,:,m));
                        [~,hat_h] = amp_2d(h_now,Phi,Psi,num,epsilon,alpha,fir0,fir1,fir2,h_fir1,h_fir2,map_level,read_level,meas_noise_flag,choice,handle_2d,stop_earlier,show_detail,noise_map,noise_read);
                        hat_x(:,:,m) = idct2(hat_h);
                    end
                    PSNR_now = psnr(x,hat_x);
                    if PSNR_now > PSNR_max
                        PSNR_max = PSNR_now;
                        Nt_max = limit(k);
                        hat_x_max = hat_x;
                    end                    
                end
                if save_flag == 1                   
                    imwrite(hat_x_max,[target_root,class,pic_name]);
                end
            else
                error('Channels must be 1 or 3');
            end
            % print
            fprintf('class %s test recons-progress : %d in %d ,PSNR = %2f at Nt = %d,class: %d in %d\n',class_name{i},j,images_num,PSNR_max,Nt_max,i-begin_class,end_class-begin_class);
            PSNRs{i_real} = [PSNRs{i_real},PSNR_max];
            Nts{i_real} = [Nts{i_real},Nt_max];
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
