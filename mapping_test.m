%% gauss+orth
clear all;

rng(11637);
N = 256;
cr = 0.75;
M = cr * N;
Phi = randn(M,N); 
Phi = sqrt(1/M) * Phi;
Phi = orth(Phi')';

%% 将Matrix映射为忆阻器矩阵电导值
L_G = 0.4;
H_G = 4;

tic;
[G_P_SM, G_N_SM,k] = G_Map_Matrix(Phi,L_G,H_G);

all_matrix = [G_P_SM,G_N_SM];
all_matrix = round(all_matrix * 100) / 100;

I_target_f2_matrix = all_matrix;

% 前256K是RP，接下来256K是RN
I_target_f2_vector = I_target_f2_matrix(:)'; 
% index代表：sort向量中各个元素在原向量中的位置
[I_target_f2_vector_sort,sort_index] = sort(I_target_f2_vector,'descend'); 
% 可以通过该方法恢复原向量:recov(sort_index,1) = I_target_f2_vector_sort; sum(I_target_f2_vector-recov)
data_summary_I_f2 = tabulate(I_target_f2_matrix(:));
%save('map_and_program/map_target/I_target_all_0404.mat',"I_target_f2_vector_sort","sort_index","I_target_f2_vector","I_target_f2_matrix","data_summary_I_f2");
save('map_and_program/map_target/test.mat',"I_target_f2_vector_sort","sort_index","I_target_f2_vector","I_target_f2_matrix","data_summary_I_f2");
% 将矩阵转换为一维向量
%vector = reshape(I_target_f2_vector_sort, 1, []);
% 将数据写入txt文件
% dlmwrite('weight_data/data1_0404.txt', vector, 'delimiter', '\t');
dlmwrite('map_and_program/weight_data/test.txt',I_target_f2_vector_sort , 'delimiter', '\t');
toc;

%% 2024-04-07 generate WM mapping targets
clc;
close all;

%load('Fig3.mat');
load('.\map_and_program\map_result\cleaned_data_all_0407.mat')
Tik_Matrix = Phi;
[column,row] = size(Tik_Matrix);
RP = zeros(size(Tik_Matrix));
RN = zeros(size(Tik_Matrix));
[RP, RN, k] = G_Map_Matrix(Tik_Matrix, 2, 20);
all_matrix = [RP,RN];
% 最终选择f2，保留两位小数作为conductance target（保留一位小数会额外引进0.05uS的误差）
all_matrix_f2 = round(all_matrix*100)/100;
I_target_f2_matrix = all_matrix_f2 .* 0.2;
I_target_f2_matrix = round(I_target_f2_matrix*100)/100;
I_target_f2_vector = I_target_f2_matrix(:);
[I_target_f2_vector_sort,sort_index] = sort(I_target_f2_vector,'descend'); % index代表：sort向量中各个元素在原向量中的位置

[processed_data,match_relaxation_current] = match_matrix(I_target_f2_vector_sort,itarget,passing_read_current,read_relaxation_current);


%%
% 可以通过该方法恢复原向量：recov(sort_index,1) = I_target_f2_vector_sort; sum(I_target_f2_vector-recov)
recov(sort_index,1) = processed_data; 
aver1=sum(I_target_f2_vector-recov)/size(I_target_f2_vector,1);

% 获取 all_matrix 的原始形状
original_size = size(all_matrix);

% 使用 reshape 函数将 I_target_f2_vector 恢复为原始形状
recovered_matrix = reshape(recov, original_size);

% 除以缩放因子，恢复原始值
recovered_matrix = recovered_matrix / 0.2;
aver2=sum(all_matrix_f2-recovered_matrix)/size(I_target_f2_vector,1);

RP1 = recovered_matrix(:,1:256);
RN1 = recovered_matrix(:,257:512);
R_M = Recover_Matrix(RP1,RN1,k);
A_recov_0 = R_M;

A_recov_set = zeros(192,256,100);
for i = 1:1:100
    
    recov(sort_index,1) = match_relaxation_current(:,i); 
    aver1=sum(I_target_f2_vector-recov)/size(I_target_f2_vector,1);

    % 获取 all_matrix 的原始形状
    original_size = size(all_matrix);

    % 使用 reshape 函数将 I_target_f2_vector 恢复为原始形状
    recovered_matrix = reshape(recov, original_size);

    % 除以缩放因子，恢复原始值
    recovered_matrix = recovered_matrix / 0.2;
    aver2=sum(all_matrix_f2-recovered_matrix)/size(I_target_f2_vector,1);

    RP1 = recovered_matrix(:,1:256);
    RN1 = recovered_matrix(:,257:512);
    R_M = Recover_Matrix(RP1,RN1,k);
    
    A_recov_set(:,:,i) = R_M;
end 

%save('.\map_and_program\map_result\processed_cleaned_data_0407.mat', 'A_recov_set','A_recov_0');


%% plot matrix 2024-4-14
%save('.\map_and_program\map_result\processed_data_plot_0415.mat', 'A_recov_set','A_recov_0','Phi','I_target_f2_matrix','recovered_matrix','');
clear all;
load('.\map_and_program\map_result\processed_data_plot_0415.mat');
R_M = A_recov_0;
% 重建前后矩阵对比
close all;
figure(1);
subplot(3,1,1);imagesc(Phi);colorbar;xticks(0:64:256);yticks(0:64:192);
title('Original matrix','Fontsize',12);
%caxis([cb1 cb2]);
subplot(3,1,2);imagesc(R_M);colorbar;xticks(0:64:256);yticks(0:64:192);
title('Mapping matrix','Fontsize',12);
%caxis([cb1 cb2]);
subplot(3,1,3);imagesc(abs((Phi - R_M)));colorbar;xticks(0:64:256);yticks(0:64:192);
title('Difference','Fontsize',12);
%caxis([cb1 cb2]);

%% 映射矩阵电导值分布 2024-4-14
figure(2);
cb1 = 0;
cb2 = 22;
original_matrix = I_target_f2_matrix/0.2;

figure(2);
subplot(3,1,1);
imagesc(original_matrix);
colorbar;xticks(0:64:512);yticks(0:64:192);
title('Original conductance','Fontsize',12);
caxis([cb1 cb2]);

subplot(3,1,2);
imagesc(recovered_matrix);
colorbar;xticks(0:64:512);yticks(0:64:192);
title('Mapping conductance','Fontsize',12);
caxis([cb1 cb2]);

subplot(3,1,3);
imagesc(abs(original_matrix - recovered_matrix));
colorbar;xticks(0:64:512);yticks(0:64:192);
title('Difference','Fontsize',12);
caxis([cb1 cb2]);


%% 映射矩阵电导值差值统计分布 2024-4-15
diff_matrix = (Phi - R_M);
diff_matrix = diff_matrix(:);

diff_conductance = (original_matrix - recovered_matrix);
diff_conductance = diff_conductance(:);

figure;
subplot(2,1,1);
histogram(diff_matrix,100,'FaceColor',"#0072BD");
xlim([-0.03,0.03]);
xlabel('element difference','FontSize',13);ylabel('frequency','FontSize',13);
title(sprintf('matrix-diff distribution with std = %.2e',std(diff_matrix(:))),'FontSize',15);
grid;
subplot(2,1,2);
histogram(diff_conductance,100,'FaceColor',"#EDB120");
xlim([-3,3]);
xlabel('conductance/uS','FontSize',13);ylabel('frequency','FontSize',13);
title(sprintf('conductance-diff distribution with std = %.2e',std(diff_conductance(:))),'FontSize',15);
grid;

%% 映射矩阵电导值统计分布 2024-4-16
figure;
subplot(2,1,2);
histogram(diff_matrix,100,'FaceColor',"#0072BD");
xlim([-0.03,0.03]);
xlabel('element difference','FontSize',13);ylabel('frequency','FontSize',13);
title(sprintf('matrix-diff distribution with std = %.2e',std(diff_matrix(:))),'FontSize',15);
grid;
subplot(2,1,1);
histogram(Phi(:),100,'FaceColor',"#EDB120");
xlim([-0.3,0.3]);
xlabel('element','FontSize',13);ylabel('frequency','FontSize',13);
title(sprintf('matrix element distribution with std = %.2e',std(Phi(:))),'FontSize',15);
grid;

%% 映射矩阵电导值变化 2024-4-15
figure;
local_x = 2;
local_y = 2;
A_change = A_recov_set(local_x,local_y,:);
A_change = A_change(:);
plot(1:1:100,A_change,'LineWidth',1.5);
hold on;
plot(1:1:100,ones(1,100)*A_recov_0(local_x,local_y),'--','LineWidth',1.5);
grid;
xlabel('read number','FontSize',13);
ylabel('element','FontSize',13);
title(sprintf('matrix element (%d,%d) change with read number',local_x,local_y),'FontSize',15);
%plot(rand100',A_change,'*','LineWidth',1.5);
legend('read','map','FontSize',10);


%% 生成1~100的一个随机序列
%rand100 = randperm(100);

%% 1-1 AMP 1D test 2024-4-11
clear all;
close all;
load ('.\save-mat\params_0401.mat');
%load ('.\map_and_program\map_result\processed_cleaned_data_0407.mat');
rand100 = randperm(100);
map_level = 0.02;
read_level = 0.02;
noise_info = [0,map_level,0,read_level;
              1,map_level,0,read_level;
              0,map_level,1,read_level;
              1,map_level,1,read_level];
% 一维仿真必加，1为无噪，4为全
noise_info_choice = noise_info(4,:);

K = 48;
h = zeros(N,1);
h(randsample(1:N,K)) = randn(1,K);
% cr = 0.75;
% M = round(N * cr);
% rng(11637);
% Phi = randn(M,N); 
% Phi = sqrt(1/M) * Phi;
% Phi = orth(Phi')';
% Psi = eye(N);
% A = Phi * Psi;
y = A * h;
theta = h;

%stop_earlier = 1;
tic;
[hat_h,~] = amp_core(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,noise_info_choice,stop_earlier,show_detail);
toc;

%relax_flag = 1;
tic;
[hat_h_test,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
toc;
NMSE = (norm(h - hat_h)^2)/(norm(h)^2);
NMSE_test = (norm(h - hat_h_test)^2)/(norm(h)^2);

figure(1);
subplot(2,1,1);
plot(1:1:N,h,'r-');
hold on;
plot(1:1:N,hat_h,'k*');
xlim([0 N]);
xticks(0:N/8:N);
legend('Original','Recons-sim','FontSize',10);
grid;
ylabel('Signal','Fontsize',12);
titleStr = sprintf('1D sparse signal reconstructed by AMP simulation with NMSE=%.2e', NMSE);
%titleStr = sprintf('1D sparse signal reconstructed by AMP');
title(titleStr,'Fontsize',12);

subplot(2,1,2);
plot(1:1:N,h,'r-');
hold on;
plot(1:1:N,hat_h_test,'b*');
xlim([0 N]);
xticks(0:N/8:N);
legend('Original','Recons-test','FontSize',10);
grid;
ylabel('Signal','Fontsize',12);
titleStr = sprintf('1D sparse signal reconstructed by AMP array-test with NMSE=%.2e', NMSE_test);

title(titleStr,'Fontsize',12);

%% 1-2 AMP 1D test 2024-4-11
close all;
nls = 0:0.005:0.10;
NMSEs = zeros(1,length(nls));
for i = 1:1:length(nls)
    map_level = nls(i);
    read_level = nls(i);
    noise_info = [1,map_level,1,read_level];
    [hat_h,~] = amp_core(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,noise_info,stop_earlier,show_detail);
   NMSEs(i) = (norm(h - hat_h)^2)/(norm(h)^2);
end

figure;
semilogy(nls,NMSEs,'-*','linewidth',1.5);
% [hat_h_test,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
% NMSE_test = (norm(h - hat_h_test)^2)/(norm(h)^2);
grid;
hold on;
plot(nls,NMSE_test*ones(1,length(nls)),'--','linewidth',1.5);
xlabel('noise level','Fontsize',12);
ylabel('NMSE','Fontsize',12);
xlim([-0.005,nls(end)+0.005]);
xticks(0:0.01:nls(end));
title('1D-AMP reconstruction NMSE-noise level curve','Fontsize',12);
legend('Recons-sim','Recons-test','FontSize',10);

%% 修改'.\save-mat\params_0401.mat'中的参数
% clear all;
% close all;
% load ('.\save-mat\params_0401.mat');
% relax_flag = 1;
% M = cr * N;
% rng(11637);
% Phi = randn(M,N); 
% Phi = sqrt(1/M) * Phi;
% Phi = orth(Phi')';
% Psi = eye(N,N);
% A = Phi * Psi;
% meas_flag = 0;
% save ('.\save-mat\params_0401.mat');
%% 2-1 AMP 2D test 2024-4-11
clear all;
close all;
load ('.\save-mat\params_0401.mat');
map_level = 0.02;
read_level = 0.02;
h = im2double(imread('.\pre-pic\bird.png'));
choice = [1,0,0,1];
tic;
[PSNR,hat_theta,hat_h] = amp_2d_gray(N,h,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
toc;

tic;
[PSNR_test,hat_theta_test,hat_h_test] = amp_2d_gray_test(h,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
toc;


%% 2-2 AMP 2D RGB test 2024-4-12
clear all;
close all;
load ('.\save-mat\params_0401.mat');
map_level = 0.02;
read_level = 0.02;
% handle_2d = 1;
h_rgb = im2double(imread('.\pre-pic\bird_RGB.png'));
choice = [1,0,0,1];

tic;
[PSNR_rgb,hat_theta_rgb,hat_h_rgb] = amp_2d_rgb(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
toc;

tic;
[PSNR_test_rgb,hat_theta_test_rgb,hat_h_test_rgb] = amp_2d_rgb_test(h_rgb,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
toc;

figure;
hold on;
subplot(1,3,1);
imshow(h_rgb);
title('Original','Fontsize',12);

subplot(1,3,2);
imshow(hat_h_rgb(:,:,:,4));
title(sprintf('nl=0.02 PSNR=%.2fdB',PSNR_rgb(4)),'Fontsize',12);

subplot(1,3,3);
imshow(hat_h_test_rgb);
title(sprintf('test PSNR=%.2fdB',PSNR_test_rgb),'Fontsize',12);

%% 2-3 AMP 2D test 2024-4-13
close all;
clear all;
load ('.\save-mat\params_0401.mat');
% handle_2d = 1;
choice = [0,0,0,1];
nls = 0:0.005:0.10;
PSNRs = zeros(1,length(nls));
for i = 1:1:length(nls)
    map_level = nls(i);
    read_level = nls(i);
    [PSNR_now,~,~] = amp_2d_rgb(N,h_rgb,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    PSNRs(i) = PSNR_now(4);
end

figure;
plot(nls,PSNRs,'-*','linewidth',1.5);
% [hat_h_test,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
% NMSE_test = (norm(h - hat_h_test)^2)/(norm(h)^2);
grid;
hold on;
plot(nls,PSNR_test_rgb*ones(1,length(nls)),'--','linewidth',1.5);
xlabel('noise level','Fontsize',12);
ylabel('PSNR','Fontsize',12);
xlim([-0.005,nls(end)+0.005]);
xticks(0:0.01:nls(end));
title('2D-AMP reconstruction PSNR-noise level curve','Fontsize',12);
legend('Recons-sim','Recons-test','FontSize',10);

%% 3-1 dataset recons test 2024-4-14
clear all;
close all;
load ('.\save-mat\params_0401.mat');
source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
target_root = '.\ImageNet\0321-5class\test-recons-val-100k-rgb\';
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;
save_flag = 1;
PSNRs_test = amp_2d_recons_test(A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,...
                                      rand100,relax_flag,meas_flag,...
                                      source_root,target_root,class_name,begin_class,end_class,save_flag);
%% 
save_flag = 0;
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb\';
PSNRs_noiseless = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);


[mu_test,sigma_test,PSNR_vec_test] = PSNR_evaluate(PSNRs_test);
[mu_noiseless,sigma_noiseless,PSNR_vec_noiseless] = PSNR_evaluate(PSNRs_noiseless);
save('.\save-mat\psnr_0414_test.mat', 'PSNRs_test','PSNR_vec_test','mu_test','sigma_test',...
                    'PSNRs_noiseless','PSNR_vec_noiseless','mu_noiseless','sigma_noiseless');


                
%% 3-2 dataset recons plot 2024-4-14
PSNR_stat(PSNR_vec_noiseless,PSNR_vec_test)

%% 
clear all;
close all;
figure(1);
ends_now = 0.09;
nls = 0:0.01:ends_now;
nets = {'ResNet18','ResNet34','ResNet50','ResNet101','ResNet152'};
%results = zeros(length(nets),length(nls_now)+1);

results = xlsread('noise_level.xlsx','C20:L22');
PSNR_mu = xlsread('noise_level.xlsx','C15:L15');
PSNR_test_rgb = xlsread('noise_level.xlsx','S15:S15');
test_results = xlsread('noise_level.xlsx','S20:S22');

figure(1);
% for i=1:1:3
%     plot(nls,results(i,:),'-*','linewidth',1.2,'DisplayName', sprintf(nets{i}));
% end
plot(nls,PSNR_mu,'-*','linewidth',1.5);
hold on;
grid;
plot(nls,PSNR_test_rgb*ones(1,length(nls)),'--','linewidth',1.5);
xlabel('noise level','Fontsize',12);
ylabel('avg-PSNR','Fontsize',12);
xlim([-0.005,nls(end)+0.005]);
xticks(0:0.01:nls(end));
title('dataset reconstruction PSNR-noise level curve','Fontsize',15);
legend('sim','test','FontSize',12);

figure(2);
c = 1;
hold on;
for i=1:1:3
    plot(nls,results(i,:),'-*','linewidth',1.2,'DisplayName', sprintf(['sim-',nets{i}]));
end
grid;

% for c=1:1:3
%     plot(nls,test_results(c)*ones(1,length(nls)),'--','linewidth',1.2,'DisplayName', sprintf(['test-',nets{c}]));
% end

plot(nls,test_results(c)*ones(1,length(nls)),'--','linewidth',1.5,'DisplayName', sprintf(['test-',nets{c}]));

hold off;
legend show;
xlim([-0.005,nls(end)+0.005]);
xticks(0:0.01:nls(end));
xlabel('noise level','Fontsize',12);
ylabel('classification accuracy','Fontsize',12);
title('AMP reconstruction accuracy-noise level curve','Fontsize',15);

%% 2% noise test
clear all;
close all;
load ('.\save-mat\params_0401.mat');
choice = [0,0,0,1];
source_root = '.\ImageNet\0321-5class\val-100k-rgb\';
target_root = '.\ImageNet\0321-5class\recons-val-100k-rgb\';
map_level = 0.02;
read_level = 0.02;
class_name = getAllSubfolders(source_root);
begin_class = 1;
end_class = 5;                         
save_flag = 0;
PSNRs_noiseless = amp_2d_recons(N,cr,orth_flag,num,epsilon,alpha,handle_2d,map_level, ...
                   read_level,choice,fir1,fir2,x_fir,trans_type,show_detail,source_root,target_root,class_name,begin_class,end_class,save_flag);


[mu_noiseless,sigma_noiseless,PSNR_vec_noiseless] = PSNR_evaluate(PSNRs_noiseless);


%% ====================== Function Below ====================== %%
%% data_wash 2024-4-1
function [G_P_SM,G_N_SM,k] = G_Map_Matrix(matrix,L_G,H_G)
    % 将Matrix映射为忆阻器矩阵电导值
    % 输出：G_P_SM 映射后的正电导矩阵
    % G_N_SM 映射后的负电导矩阵
    % 输入：matrix 生成的感知矩阵
    % L_G 映射电导的下限
    % H_G 映射电导的上限e2
    
    [m,n] = size(matrix);
    Value_Max = max(max(abs(matrix)));        % 寻找Matrix中的绝对值最大值
    %Value_Max = 1;                           % 对于DFT矩阵，最大值可以都默认是1；
    Value_Min = 0;                            % Matrix中的绝对值最小值直接设置为0，这样整体是一个线性映射
    
    k = (H_G-L_G)/(Value_Max-Value_Min);
    
    G_P_SM = double(zeros(m, n));
    G_N_SM = double(zeros(m, n));
    
    G_general = matrix * k;
    
    for col = 1 : n
        for row = 1 : m
            if G_general(row,col) >= 0
                % 电导最小值式映射
%                 G_N_SM(row,col) = L_G;
%                 G_P_SM(row,col) = G_general(row,col) + L_G;

                % 电导最大值式映射
                G_N_SM(row,col) = H_G-G_general(row,col);
                G_P_SM(row,col) = H_G;
            else
                % 电导最小值式映射
%                 G_P_SM(row,col) = L_G;
%                 G_N_SM(row,col) = -G_general(row,col) + L_G;

                % 电导最大值式映射
                G_N_SM(row,col) = H_G;
                G_P_SM(row,col) = H_G+G_general(row,col);
            end
        end
    end
end

function [matrix_recovered] = Recover_Matrix(G_P_SM,G_N_SM,k)
    %[m,n] = size(G_P_SM);
    %matrix_recovered = zeros(m,n);
    matrix_recovered = (G_P_SM - G_N_SM) / k;
end

function [processed_data,matching_read_relaxation_current] = match_matrix(data,itarget,passing_read_current,read_relaxation_current)
% 初始化一个新的数组来存储处理后的数据
processed_data = zeros(size(data));
matching_read_relaxation_current = zeros(size(data,1),100);
% 获取数据的长度
data_length = length(data);

% 开始计时
tic;

for i = 1:data_length
    % 对数据进行处理
    [processed_data(i),matching_read_relaxation_current(i,:)] = find_matching_values(itarget,passing_read_current,read_relaxation_current,data(i));

    % 每处理1000个数据，打印一次已经过去的时间
    if mod(i, 1000) == 0
        elapsed_time = toc;
        fprintf('Processed %d/%d data points. Elapsed time is %f seconds.\n', i, data_length, elapsed_time);

        % 重新开始计时
        tic;
    end
end

end

function [chosen_passing_read_current, chosen_read_relaxation_current] = find_matching_values(itarget, passing_read_current, read_relaxation_current, input_number)
  % 找出所有与输入值匹配的索引
  matching_indices = find(itarget == input_number);

  % 如果没有找到匹配的值，返回空的结果
  if isempty(matching_indices)
    chosen_passing_read_current = [];
    chosen_read_relaxation_current = [];
    return;
  end

  % 随机选择一个索引
  chosen_index = randsample(matching_indices, 1);

  % 返回对应的passing_read_current和read_relaxation_current
  chosen_passing_read_current = passing_read_current(chosen_index);
  chosen_read_relaxation_current = read_relaxation_current(chosen_index, :);
end

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
        
%         read deviation     
        if read_noise_flag
            noise = max(abs(A(:)))*read_noise_level*randn(M,N);
            A_noise = A_map_noise + noise;
        else
            A_noise = A_map_noise;
        end
        
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
                    hat_theta(i,:,noise_tag) = hat_theta_elem';
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
    % legend('noiseless','map+read noise','FontSize',10);
    legend('sim','test','FontSize',10);
    grid;
end

%% AMP real-test 2024-4-11
%1D test
function [x,z] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag)
    %y: M*1 signal after measurement
    %num: max iteration num
    %epsilon: min iteration step
    %alpha: usually alpha=1 for this fucntion
    %theta: ground truth for sparse vector to be reconstructed]
    %stop_earlier: no need for run all num iteration...
    %...if less than epsilon, break.
    
    %x: N*1 sparse signal reconstruction result
    %z: M*1 residual(during iteration)
    %MSE/NMSE: MSE/NMSE change in the iteration process
    
    MSE = [];
    NMSE = [];
    A_example = A_recov_set(:,:,1);
    [M,N] = size(A_example);
    x0 = zeros(N,1);
    z0 = y;
    
    A_noise = A_recov_0;
    %iteration
    % x0,z0→x,z
    for t = 1:1:num
        % equal_index = mod(t - 1, num) + 1;
        if relax_flag == 1
            equal_index = t;
            fetch_index = rand100(equal_index);
            A_noise = A_recov_set(:,:,fetch_index);
        end
        
        lambda0 = alpha*norm(z0,2) / sqrt(M);
        
        b0 = sum(x0(:)~=0) / M;   %AMP append element        

        % [~,r0] = fir_self(A_noise' * z0, fir1);
        r0 = A_noise' * z0;
        
        x = amp_eta_t(r0 + x0, lambda0);
        
        % [~,s] = fir_self(A_noise * x, fir2);
        s = A_noise * x;
        
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
    
   
end

% 2D single rea1 map test
function [hat_theta] = amp_2d_test(theta,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,show_detail,stop_earlier,rand100,relax_flag,meas_flag)
    
    
    [~,N] = size(A);
    hat_theta = zeros(N,N);
    
    if handle_2d == 0
        % column-wise handle

        for i = 1:1:N
            theta_now = theta(:,i);
            if meas_flag == 0
                y = A * theta_now;
            else
                y = A_recov_0 * theta_now;
            end
            % theta_now = theta(i,:)';
            % x = Psi * theta_now;
            % y = Phi * x;     
            % A = Phi * Psi;                    
            [hat_theta_elem,~] = amp_core_test(y,A_recov_set,A_recov_0,theta_now,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);              
            hat_theta(:,i) = hat_theta_elem;
        end
    elseif handle_2d == 1
        % row-wise handle
        for i = 1:1:N
            theta_now = theta(i,:)';
            if meas_flag == 0
                y = A * theta_now;
            else
                y = A_recov_0 * theta_now;
            end
            % theta_now = theta(i,:)';
            % x = Psi * theta_now;
            % y = Phi * x;     
            % A = Phi * Psi;                    
            [hat_theta_elem,~] = amp_core_test(y,A_recov_set,A_recov_0,theta_now,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);              
            hat_theta(i,:) = hat_theta_elem';
        end       
    else
        error('This handle has not been developed.');
    end

end

% 2D single channel picture test
function [PSNR,hat_theta,hat_h] = amp_2d_gray_test(h,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag)
    
    [~,N] = size(A);
        
    h = pic2double(h,N);
    
    % record result
    % hat_h = zeros(N,N,4);
    % PSNR = zeros(1,4);
    
    %trans_type
    if trans_type == 'dct'
        % dct sparse
        theta = dct2(h);
        hat_theta = amp_2d_test(theta,A,A_recov_set,A_recov_0,num,epsilon,alpha,...
                    handle_2d,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
        % record PSNR
        hat_h = idct2(hat_theta);
        PSNR = psnr(hat_h,h);
    elseif trans_type == 'non'
        % origin sparse
        theta = h;
        hat_theta = amp_2d_test(theta,A,A_recov_set,A_recov_0,num,epsilon,alpha,...
                    handle_2d,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
        % record PSNR
        hat_h = idct2(hat_theta);
        PSNR = psnr(hat_h,h);
    else
        error('This trans_type has not been developed.');
    end
    
    % show PSNR detail
    formatspec_tot = '实测重构PSNR = %4f \n';
    if(show_detail==1)
        fprintf(formatspec_tot,PSNR);
    end   
end

% 2D 3-channel RGB picture picture test
function [PSNR,hat_theta,hat_h] = amp_2d_rgb_test(h,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag)
    % need to be rgb
    [~,N,color] = size(h);
    if(color~=3)
        error('Picture Must be RGB three channel.')
    end
    % record result
    hat_theta = zeros(N,N,3);
    hat_h = zeros(N,N,3);
    PSNR_now = zeros(1,3);
    % 3 channels reconstruction
    for i = 1:1:3
        h_now = h(:,:,i);
        [PSNR_now(:,i),hat_theta(:,:,i),hat_h(:,:,i)] = amp_2d_gray_test(h_now,A,A_recov_set,A_recov_0,num,...
                                                           epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
end

function [PSNRs] = amp_2d_recons_test(A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,...
                                      rand100,relax_flag,meas_flag,...
                                      source_root,target_root,class_name,begin_class,end_class,save_flag)
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
        
        i_real = i - begin_class + 1;
        PSNRs{i_real} = [];
        for j = 1:1:images_num
            % read
            pic_name = images(j).name;   
            h = imread([source_root,class,pic_name]);
            [~,~,color] = size(h);
            % recons and save
            if color == 1
                % [PSNR,~,hat_h] = amp_2d_gray(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
                [PSNR_now,~,hat_h] = amp_2d_gray_test(h,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
                if save_flag == 1                   
                    imwrite(hat_h,[target_root,class,pic_name]);
                end                
            elseif color == 3
                % [PSNR,~,hat_h] = amp_2d_rgb(N,h,compress_rate,orth_flag,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
                [PSNR_now,~,hat_h] = amp_2d_rgb_test(h,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
                if save_flag == 1                   
                    imwrite(hat_h,[target_root,class,pic_name]);
                end
            else
                error('Channels must be 1 or 3');
            end
            % print
            fprintf('class %s test recons-progress : %d in %d ,PSNR = %2f,class: %d in %d\n',class_name{i},j,images_num,PSNR_now,i-begin_class,end_class-begin_class);
            PSNRs{i_real} = [PSNRs{i_real},PSNR_now];
        end
        toc;
    end

end

