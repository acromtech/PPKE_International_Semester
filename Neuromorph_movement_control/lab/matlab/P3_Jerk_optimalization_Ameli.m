%% Jerk optimum movement
clear all;
close all;
% and Inverse kinematics problem
% 2024.03.19

segment_lengths = [0.3,0.25,0.1]; % length of segments

timesteps_num = 100;

angles = [-pi/2, pi/4, pi/2]; % Initial angles


%% Solve IK problem 

angles = inverse_kin_and_smoothness(angles, [-0.1,0.2], segment_lengths, timesteps_num);
angles = inverse_kin_and_smoothness(angles, [0.5, 0.4], segment_lengths, timesteps_num);
angles = inverse_kin_and_smoothness(angles, [0.3, -0.4], segment_lengths, timesteps_num);

%% Define IK solver function


function final_a = inverse_kin_and_smoothness(init_angles, target, seg_lens, ts_num)

    assert(sqrt(target(1)^2+target(2)^2) <= sum(seg_lens));     %out of reach error message
    
    %% Define placeholders
    
    alphas = zeros(3,ts_num);
    alphas(:,1) = init_angles;
    
    a1 = alphas(1,1);
    a12 = a1 + alphas(2,1);
    a123 = a12 + alphas(3,1);
    
    E1_inverse = zeros(2, ts_num);
    E2_inverse = zeros(2, ts_num);
    E3_inverse = zeros(2, ts_num);
    
    E1_inverse(:,1) = [seg_lens(1)*cos(a1); seg_lens(1)*sin(a1)];
    E2_inverse(:,1) = [E1_inverse(1,1) + seg_lens(2)*cos(a12); E1_inverse(2,1)+seg_lens(2)*sin(a12)];
    E3_inverse(:,1) = [E2_inverse(1,1) + seg_lens(3)*cos(a123); E2_inverse(2,1)+seg_lens(3)*sin(a123)];
    
    target_difference = [(target(1)-E3_inverse(1,1))/ts_num;(target(2)-E3_inverse(2,1))/ts_num];
    
    for i= 1:ts_num-1
        
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
    
    final_a = alphas(:,ts_num);
    
            
    va1 = alphas(1,:);
    va12 = va1 + alphas(2,:);
    va123 = va12 + alphas(3,:);
    
    
    E1_inverse = [seg_lens(1).*cos(va1); seg_lens(1).*sin(va1)];
    E2_inverse = [E1_inverse(1,:) + seg_lens(2).*cos(va12); E1_inverse(2,:)+seg_lens(2).*sin(va12)];
    E3_inverse = [E2_inverse(1,:) + seg_lens(3).*cos(va123); E2_inverse(2,:)+seg_lens(3).*sin(va123)];
    
    tau = linspace(0,1,ts_num);
    beta = zeros(3,ts_num);  % angles by optimal  jerk 
    
    beta0 = alphas(:,1);    %defined as an input
    betaT = alphas(:,end);  % i know it 'cuz inverse kinematics calculation
  
    beta = beta0+(beta0-betaT)*(15.*tau.^4-6.*tau.^5-10.*tau.^3); %using vector, ergo dont have to calculate separetly for wrist, elbow or shoulder
    
    b1 = beta(1,:);
    b12 = b1 + beta(2,:);
    b123 = b12 + beta(3,:);
    
    
    E1_smooth_jerk = [seg_lens(1).*cos(b1); seg_lens(1).*sin(b1)];
    E2_smooth_jerk = [E1_smooth_jerk(1,:) + seg_lens(2).*cos(b12); E1_smooth_jerk(2,:)+seg_lens(2).*sin(b12)];
    E3_smooth_jerk = [E2_smooth_jerk(1,:) + seg_lens(3).*cos(b123); E2_smooth_jerk(2,:)+seg_lens(3).*sin(b123)];
    
    
    figure(1);
    
    for i = 1:ts_num
       
        arm_inv = plot([0,E1_inverse(1,i),E2_inverse(1,i),E3_inverse(1,i)],[0,E1_inverse(2,i),...
            E2_inverse(2,i),E3_inverse(2,i)],'k-','LineWidth',1.2);
        arm_jerk = plot([0,E1_smooth_jerk(1,i),E2_smooth_jerk(1,i),E3_smooth_jerk(1,i)],...
            [0,E1_smooth_jerk(2,i),E2_smooth_jerk(2,i),E3_smooth_jerk(2,i)],'Color','#D95319','LineWidth',1.2);
        hold on; grid on;
        
        targ_mark = plot(target(1),target(2),'r.','Markersize', 25);
        xlim([-1,1]);
        ylim([-1,1]);
        legend('iterative','optimal\_jerk','target');
        
        drawnow;
        pause(0.001 + 100/(20*ts_num));
        delete(arm_inv);
        delete(arm_jerk);
        delete(targ_mark);


        
    end

   % elbow angel change during movement
   % figure(23);
   % plot(va12);hold on;
   % plot(b12)
   % waitforbuttonpress;
   % 
    


end


