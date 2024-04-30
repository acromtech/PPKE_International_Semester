close all;
clear all;

fname = 'kin1.xlsx';
test_ratio = 0.8;
n = 2;

data = xlsread(fname);
data = data(3:end,3:end);   %first 2 colum time, first 2 row is negotiable data
data(isnan(data)) = 0;      %data cleaning

Num_total = length(data);       
Num_test = round(Num_total*test_ratio);     %testing, how much percent is used to testing

data_calibration = data(3:Num_total-Num_test,:);    %teaching the motion with the left 20% of the data
data_test = data(Num_total-Num_test+1:Num_total,:);

CovData = cov(data_calibration); %calculating cov. matrix
[V,D] = eigs(CovData,n);        %the first 2 biggest eigen vectors of the matrix...
                                   %V are vectors, D are eigen values
PM = V(:,1:n);                  %choosing the vectors what i want to use
% We check that how good is PM for control purposes
% projection matrix, this will be used to a projection from the 8D Ňstate...
% space to a Ň2D external space.

[VV,DD] = eigs(CovData, width(CovData));

% trace(DD); % the sum of all the eigen values.
%tace(D); the sum of the n largest eigen values.

ratio = trace(D)/trace(DD); %+sum the diagonal elements of D &DD matrices

C = zeros([Num_test,n]);

for k=1:Num_test
    
    S(k,:) = [data_test(k,:)]; %reading out by rows from test, like sensors
 %   S(k,:) = [data_test(k,:)]; %reading out by rows from test, like sensors
    C(k,:) = S(k,:)*PM;         %control signal, what you see on screen
    
    if n == 2
        p = plot(C(k,1), C(k,2),'b.','Markersize',20);

        hold on;

        xlabel('x');
        ylabel('y');
        pause(0.03);

    elseif n == 3

        p = plot3(C(k,1), C(k,2), C(k,3),'b.','Markersize',10);
        hold on;

        % 
        % if fname == 'kin1.xlsx'
        %     %axis([-5560 -3500 -600 250 -800 600]);
        % elseif fname == 'kin2.xlsx'
        %     %axis([1200 1800 -2000 -1800 420 500]);
        % elseif fname == 'emg4.xlsx'   
        %     axis([350 460 0 10 -1 1.5]);
        % end
        xlabel('x');
        ylabel('y');
        zlabel('z');
        pause(0.01);

    end

end
