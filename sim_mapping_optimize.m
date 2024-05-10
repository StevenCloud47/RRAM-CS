%% 截断较小值 2024-4-17 按列分割
clear all;
close all;

h_pic = im2double(imread('.\pre-pic\bird.png'));
hat_theta_test = zeros(N,N);

[PSNR_test,~,~] = amp_2d_gray_test(h_pic,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,...
                              trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
trunc_flag = 1;
tic;
h_dct = dct2(h_pic);
limit = [0.1:0.2:0.9, 1,5,10,20,50,100];
% limit = [0.01,0.1,1,10,100,1000];
hat_h_all = zeros(N,N,length(limit));
PSNR = zeros(1,length(limit));
for j = 1:1:length(limit)
    for i = 1:1:N
        h = h_dct(:,i);
        y = A * h;
        theta = h;
        [hat_theta_now,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);

        %NMSE_test = (norm(h - hat_theta_now)^2)/(norm(h)^2);
        diff_test = (theta - hat_theta_now);
        diff_test_norm = diff_test ./ theta;
        truncated_index = find(diff_test_norm > limit(j));

        if trunc_flag == 1
            hat_theta_now(truncated_index) = 0;
            hat_theta_test(:,i) = hat_theta_now;
        else
            hat_theta_test(:,i) = hat_theta_now;
        end
    end
    hat_h_test = idct2(hat_theta_test);
    PSNR(j) = psnr(hat_h_test,h_pic);
    hat_h_all(:,:,j) = hat_h_test;
end
toc;

figure;
semilogx(limit,PSNR,'*-','LineWidth',1.5);
grid;
hold on;
plot(limit,PSNR_test*ones(1,length(limit)),'--','LineWidth',1.5);
xlabel('truncated threshold','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('truncated','truncate-less','FontSize',10);
%xlim([0 10.1]);xticks(0:1:10);
title('2D AMP reconstruction PSNR-truncated threshold curve','FontSize',12);


%% 截断较小值 2024-4-17 无需原值
clear all;
close all;
load ('.\save-mat\params_0401.mat');

%h_pic = im2double(imread('.\pre-pic\house.bmp'));
h_pic = im2double(imread('.\pre-pic\6.png'));
h_pic = pic2double(h_pic,N);
hat_theta_test = zeros(N,N);

[PSNR_soft,~,~] = amp_2d_gray(N,A,h_pic,cr,num,epsilon,alpha,handle_2d,...
                              map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);

[PSNR_test,~,~] = amp_2d_gray_test(h_pic,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,...
                              trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
trunc_flag = 0;
tic;
h_dct = dct2(h_pic);
limit = 16:8:256;
hat_h_all = zeros(N,N,length(limit));
PSNR = zeros(1,length(limit));
for j = 1:1:length(limit)
    x_fir = [ones(limit(j),1);zeros(N-limit(j),1)];
    for i = 1:1:N
        h = h_dct(:,i);
        y = A * h;
        theta = h;
        [hat_theta_now,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
        hat_theta_test(:,i) = hat_theta_now .* x_fir;
%         hat_theta_now = hat_theta_now .* x_fir;
%         diff_test = (theta - hat_theta_now);
%         diff_test_norm = diff_test ./ theta;
%         truncated_index = find(diff_test_norm > 1);
% 
%         if trunc_flag == 1
%             hat_theta_now(truncated_index) = 0;
%             hat_theta_test(:,i) = hat_theta_now;
%         else
%             hat_theta_test(:,i) = hat_theta_now;
%         end
    end
    hat_h_test = idct2(hat_theta_test);
    PSNR(j) = psnr(hat_h_test,h_pic);
    hat_h_all(:,:,j) = hat_h_test;
end
toc;

figure;
plot(limit,PSNR,'*-','LineWidth',1.5);
grid;
hold on;
plot(limit,PSNR_test*ones(1,length(limit)),'--','LineWidth',1.5);
xlabel('simple truncate number','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('truncated','truncate-less','FontSize',10);
xlim([8 264]);xticks(16:16:256);
title('2D AMP reconstruction PSNR-simple truncate number curve','FontSize',12);
[val,find] = max(PSNR);
val
limit(find)

%% 过渡截断 2024-4-20 无需原值
clear all;
close all;
load ('.\save-mat\params_0401.mat');

%h_pic = im2double(imread('.\pre-pic\house.bmp'));
h_pic = im2double(imread('.\pre-pic\bird.png'));
h_pic = pic2double(h_pic,N);
hat_theta_test = zeros(N,N);

[PSNR_soft,~,~] = amp_2d_gray(N,Phi,h_pic,cr,num,epsilon,alpha,handle_2d,...
                              map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
[PSNR_test,~,~] = amp_2d_gray_test(h_pic,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,...
                              trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);

trunc_flag = 0;
tic;
h_dct = dct2(h_pic);
N1 = 208;
limit = 8:8:N-N1;
hat_h_all = zeros(N,N,length(limit));
PSNR = zeros(1,length(limit));
for j = 1:1:length(limit)
    %x_fir = [ones(limit(j),1);zeros(N-limit(j),1)];
    x_fir = create_trunc(N1,N1+limit(j),N);
    for i = 1:1:N
        h = h_dct(:,i);
        y = A * h;
        theta = h;
        [hat_theta_now,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
        hat_theta_test(:,i) = hat_theta_now .* x_fir;
%         hat_theta_now = hat_theta_now .* x_fir;
%         diff_test = (theta - hat_theta_now);
%         diff_test_norm = diff_test ./ theta;
%         truncated_index = find(diff_test_norm > 1);
% 
%         if trunc_flag == 1
%             hat_theta_now(truncated_index) = 0;
%             hat_theta_test(:,i) = hat_theta_now;
%         else
%             hat_theta_test(:,i) = hat_theta_now;
%         end
    end
    hat_h_test = idct2(hat_theta_test);
    PSNR(j) = psnr(hat_h_test,h_pic);
    hat_h_all(:,:,j) = hat_h_test;
end
toc;

figure;
plot(N1+limit,PSNR,'*-','LineWidth',1.5);
grid;
hold on;
plot(N1+limit,PSNR_test*ones(1,length(limit)),'--','LineWidth',1.5);
xlabel('N_2','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('truncated','truncate-less','FontSize',10);
xlim([N1 264]);xticks(N1:16:N);
title(sprintf('2D AMP reconstruction with N_1=%d',N1),'FontSize',12);
[val,find] = max(PSNR);
val
limit(find)+N1

%% 固定阈值截断 2024-4-24 无需原值
clear all;
close all;
load ('.\save-mat\params_0401.mat');

%h_pic = im2double(imread('.\pre-pic\house.bmp'));
h_pic = im2double(imread('.\pre-pic\bird.png'));
h_pic = pic2double(h_pic,N);
hat_theta_test = zeros(N,N);

[PSNR_soft,~,~] = amp_2d_gray(N,Phi,h_pic,cr,num,epsilon,alpha,handle_2d,...
                              map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
[PSNR_test,~,~] = amp_2d_gray_test(h_pic,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,...
                              trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);

trunc_flag = 0;
tic;
h_dct = dct2(h_pic);
N1 = 208;
limit = 8:8:N-N1;
hat_h_all = zeros(N,N,length(limit));
PSNR = zeros(1,length(limit));
for j = 1:1:length(limit)
    %x_fir = [ones(limit(j),1);zeros(N-limit(j),1)];
    x_fir = create_trunc(N1,N1+limit(j),N);
    for i = 1:1:N
        h = h_dct(:,i);
        y = A * h;
        theta = h;
        [hat_theta_now,~] = amp_core_test(y,A_recov_set,A_recov_0,theta,num,epsilon,alpha,show_detail,stop_earlier,rand100,relax_flag);
        hat_theta_test(:,i) = hat_theta_now .* x_fir;
%         hat_theta_now = hat_theta_now .* x_fir;
%         diff_test = (theta - hat_theta_now);
%         diff_test_norm = diff_test ./ theta;
%         truncated_index = find(diff_test_norm > 1);
% 
%         if trunc_flag == 1
%             hat_theta_now(truncated_index) = 0;
%             hat_theta_test(:,i) = hat_theta_now;
%         else
%             hat_theta_test(:,i) = hat_theta_now;
%         end
    end
    hat_h_test = idct2(hat_theta_test);
    PSNR(j) = psnr(hat_h_test,h_pic);
    hat_h_all(:,:,j) = hat_h_test;
end
toc;

figure;
plot(N1+limit,PSNR,'*-','LineWidth',1.5);
grid;
hold on;
plot(N1+limit,PSNR_test*ones(1,length(limit)),'--','LineWidth',1.5);
xlabel('N_2','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('truncated','truncate-less','FontSize',10);
xlim([N1 264]);xticks(N1:16:N);
title(sprintf('2D AMP reconstruction with N_1=%d',N1),'FontSize',12);
[val,find] = max(PSNR);
val
limit(find)+N1


%% 0/1测量矩阵 2024-4-21
clear all;
close all;
load ('.\save-mat\params_0401.mat');
%h_pic = im2double(imread('.\pre-pic\house.bmp'));
h_pic = im2double(imread('.\pre-pic\bird.png'));
h_pic = pic2double(h_pic,N);
hat_theta_test = zeros(N,N);

[PSNR_Gauss,~,~] = amp_2d_gray(N,A,h_pic,cr,num,epsilon,alpha,handle_2d,...
                              map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
PSNR_Gauss = PSNR_Gauss(1);

[PSNR_test,~,~] = amp_2d_gray_test(h_pic,A,A_recov_set,A_recov_0,num,epsilon,alpha,handle_2d,...
                              trans_type,show_detail,stop_earlier,rand100,relax_flag,meas_flag);
tic;

limit = 0:0.01:0.01;
hat_h_all = zeros(N,N,length(limit));
PSNR = zeros(1,length(limit));
choice = [0,0,0,1];

orth_flag = 1;
M = cr * N;
n_1 = 64;
n_2 = 128;
K = n_1 * n_2;
A1 = one_zero_matrix(M,N,K,orth_flag);

for j = 1:1:length(limit)
    map_level = limit(j);
    read_level = limit(j);
    [PSNR_now,~,hat_h_now] = amp_2d_gray(N,A1,h_pic,cr,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    PSNR(j) = PSNR_now(4);
    hat_h_all(:,:,j) = hat_h_now(:,:,4);
end
toc;

%%
figure;
plot(limit,PSNR,'*-','LineWidth',1.5);
grid;
hold on;
plot(limit,PSNR_test*ones(1,length(limit)),'--','LineWidth',1.5);
xlabel('noise level','FontSize',12);
ylabel('PSNR/dB','FontSize',12);
legend('test-Gauss','sim-0/1','FontSize',10);
xlim([-0.01 0.11]);xticks(0:0.01:0.10);
title(sprintf('2D AMP reconstruction PSNR-0/1 measurement matrix with %d×%d 1',n1,n2),'FontSize',12);


%% column-wise rationality argument 2024-4-23
clear all;
close all;
load ('.\save-mat\params_0401.mat');

%h_pic = im2double(imread('.\pre-pic\house.bmp'));
h_pic = im2double(imread('.\pre-pic\bird2.png'));

% D1 = dctmtx(N);
% h_dct1 = D1 * h_pic * D1';
% h_dct2 = dct2(h_pic);
% sum(sum(abs(h_dct2 - h_dct1)))
% 
% pic_col = h_pic(:,32);
% sum(abs(D1 * pic_col - dct(pic_col)))
map_level = 0.02;
read_level = 0.02;
noise_info = [0,map_level,0,read_level;
              1,map_level,0,read_level;
              0,map_level,1,read_level;
              1,map_level,1,read_level];

Psi = dctmtx(N);
%Phi = A;
Phi = SparseRandomMtx(cr*N,N,32);
%Phi = orth(Phi')';
A = Phi * Psi';
hat_h_pic = zeros(N,N);
tic;
for i = 1:1:N
    h = h_pic(:,i);
    y = Phi * h;
    theta = Psi * h;
    [hat_theta,~] = amp_core(y,A,theta,num,epsilon,alpha,fir1,fir2,x_fir,noise_info(1,:),1,show_detail);
    hat_h = Psi' * hat_theta;
    hat_h_pic(:,i) = hat_h;
    fprintf('%d in %d\n',i,N);
end
toc;
psnr(h_pic,hat_h_pic)
coher(Phi,Psi')
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

%% trans_trunc 2024-4-20
function array = create_trunc(a,b,N)
    % 检查输入参数的有效性
    if a >= b || a < 1 || b > N
        error('Invalid inputs. Ensure that 1 <= a < b <= 256.');
    end

    % 创建一个长度为N的数组
    array = zeros(N, 1);

    % 在位置1到a的元素值设为1
    array(1:a) = 1;

    % 在位置a+1到b的元素值线性降低到0
    array(a+1:b) = linspace(1, 0, b-a);

    % 在位置b+1到N的元素值已经为0，因此无需额外设置
end


%% 0-1 meas-matrix 2024-4-21
function matrix = one_zero_matrix(M,N,K,orth_flag)
    % 计算矩阵中的总元素数
    total_elements = M * N;

    % 创建一个全零矩阵
    matrix = zeros(total_elements, 1);

    % 随机选择 K 个索引
    rng(11637);
    indices = randperm(total_elements, K);

    % 将这些索引对应的元素设为 1
    matrix(indices) = 1;

    % 重新调整矩阵的形状
    matrix = reshape(matrix, [M, N]);
    
    if orth_flag == 1
        matrix = orth(matrix')';
    end
end

function Phi = SparseRandomMtx( M,N,d )
%SparseRandomMtx Summary of this function goes here
%   Generate SparseRandom matrix 
%   M -- RowNumber
%   N -- ColumnNumber
%   d -- The number of '1' in every column,d<M 
%   Phi -- The SparseRandom matrix

% Generate SparseRandom matrix   
    Phi = zeros(M,N);
    for ii = 1:N
        ColIdx = randperm(M);
        Phi(ColIdx(1:d),ii) = 1;

    end
end

function incoh = coher(Phi,Psi)
    [M,N] = size(Phi);
    cohers_mat = zeros(M,N);
    for i = 1:1:M
        for j = 1:1:N
            cohers_mat(i,j) = abs(Phi(i,:) * Psi(j,:)');
        end
    end
    incoh = max(max(cohers_mat));
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
function [PSNR,hat_theta,hat_h] = amp_2d_gray(N,Phi,h,compress_rate,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail)
    
    h = pic2double(h,N);
    M = round(N * compress_rate); 
    
%     %Gauss
%     rng(11637);
%     Phi = randn(M,N); 
%     Phi = sqrt(1/M) * Phi;
%     if orth_flag == 1
%         Phi = orth(Phi')';
%     end
%     
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
function [PSNR,hat_theta,hat_h] = amp_2d_rgb(N,Phi,h,compress_rate,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail)
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
        [PSNR_now(:,i,:),hat_theta(:,:,i,:),hat_h(:,:,i,:)] = amp_2d_gray(N,Phi,h_now,compress_rate,num,epsilon,alpha,handle_2d,map_level,read_level,choice,fir1,fir2,x_fir,trans_type,show_detail);
    end
    % PSNR for rgb: single average
    PSNR = mean(PSNR_now,2);
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

