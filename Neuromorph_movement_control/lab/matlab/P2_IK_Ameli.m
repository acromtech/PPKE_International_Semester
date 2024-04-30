% Inverse kinematics problem
% 2024. 03. 13.

clear all;
close all;

segment_lengths = [0.3,0.25,0.1]; % 3 segment from shoulder to fingertip

timesteps_num = 100; %want to reach the target in 100 steps

angles = [-pi/2, pi/4, pi/2]; % Initial angles, using pi -> function sin/cos


%% Solve IK problem 
%this is the main task, give target to reach

angles = inverse_kinematics(angles, [0.1,0.3], segment_lengths, timesteps_num);
angles = inverse_kinematics(angles, [0.5, 0.4], segment_lengths, timesteps_num);
angles = inverse_kinematics(angles, [0.3, -0.4], segment_lengths, timesteps_num);

%% Define IK solver function


function final_a = inverse_kinematics(init_angles, target, seg_lens, ts_num)

    assert(sqrt(target(1)^2+target(2)^2) <= sum(seg_lens)); %if our point out of our radius
    
    %% Define placeholders
    
    alphas = zeros(3,ts_num); % for 3 joint angles, for the neumber of steps
    alphas(:,1) = init_angles;
    
    a1 = alphas(1,1);           %calculating for directkinematics
    a12 = a1 + alphas(2,1);
    a123 = a12 + alphas(3,1);
    
    E1 = zeros(2, ts_num);      %place holder for  results of DK equations
    E2 = zeros(2, ts_num);
    E3 = zeros(2, ts_num);
    
    E1(:,1) = [seg_lens(1)*cos(a1); seg_lens(1)*sin(a1)];
    E2(:,1) = [E1(1,1) + seg_lens(2)*cos(a12); E1(2,1)+seg_lens(2)*sin(a12)];
    E3(:,1) = [E2(1,1) + seg_lens(3)*cos(a123); E2(2,1)+seg_lens(3)*sin(a123)];
    
    target_difference = [(target(1)-E3(1,1))/ts_num;(target(2)-E3(2,1))/ts_num];
    
    for i= 1:ts_num
        
            a1 = alphas(1,i);
            a12 = a1 + alphas(2,i);
            a123 = a12 + alphas(3,i);
            
             
             J = [-seg_lens(1)*sin(a1) - seg_lens(2)*sin(a12) - seg_lens(3)*sin(a123), ...
                 - seg_lens(2)*sin(a12) - seg_lens(3)*sin(a123), ...
                 - seg_lens(3)*sin(a123);
                 seg_lens(1)*cos(a1) + seg_lens(2)*cos(a12) + seg_lens(3)*cos(a123), ...
                 seg_lens(2)*cos(a12) + seg_lens(3)*cos(a123), ...
                 seg_lens(3)*cos(a123)];
             
             delta_a = pinv(J)*target_difference;
             alphas(:,i+1) = alphas(:,i) + delta_a;
            
    end
    
    final_a = alphas(:,ts_num); %return value 
    
    %going forward with DK
            
    va1 = alphas(1,:);
    va12 = va1 + alphas(2,:);
    va123 = va12 + alphas(3,:);
    
    
    E1 = [seg_lens(1).*cos(va1); seg_lens(1).*sin(va1)];
    E2 = [E1(1,:) + seg_lens(2).*cos(va12); E1(2,:)+seg_lens(2).*sin(va12)];
    E3 = [E2(1,:) + seg_lens(3).*cos(va123); E2(2,:)+seg_lens(3).*sin(va123)];
    
    
    figure(1);
    
    for i = 1:ts_num
       
        arm = plot([0,E1(1,i),E2(1,i),E3(1,i)],[0,E1(2,i),E2(2,i),E3(2,i)],'k-');
        hold on; grid on;
        
        targ_mark = plot(target(1),target(2),'r.','Markersize', 5);
        xlim([-1,1]);
        ylim([-1,1]);
        
        drawnow;
        pause(0.001 + 100/(20*ts_num));
        delete(arm);
        delete(targ_mark);
        
    end
    
    


end