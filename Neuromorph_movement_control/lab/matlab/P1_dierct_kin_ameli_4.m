%% Practice 2024.03.13.
% *Direct kinematic* 

L1 = 0.3; % Upper arm length
L2 = 0.25; % Lower arm
L3 = 0.1;   % Hand
L4 = 0.66; %pencil

t = linspace(0,3,300); % Time 0s -> 3s;

a1 = 2*t;       % Shoulder angle
a2 = 3*t.^2;    % Elbow angle
a3 = t;         % Wrist angle
a4 = t/2;       %pencil and wirst angle

a12 = a1 + a2;          %calculate the summed angles for the next step
a123 = a12 + a3;
a1234 = a123 + a4;

%% Compute vertex locations
E1 = [L1.*cos(a1); L1.*sin(a1)];   %coordinates of the elbow
E2 = [E1(1,:)+ L2.*cos(a12); E1(2,:) + L2.*sin(a12)];   %coordinates of the wrist
E3 = [E2(1,:) + L3.*cos(a123); E2(2,:) + L3.*sin(a123)]; %coordinates of the finger tip
E4 = [E3(1,:) + L4.*cos(a1234); E3(2,:) + L4.*sin(a1234)]; %coordinates: end of the pencil


%% Plot the arm movemets

for i = 1:300
    plot([0, E1(1,i), E2(1,i),E3(1,i), E4(1,i)], [0, E1(2,i), E2(2,i),E3(2,i),E4(2,i)])
    axis([-0.5,0.5,-0.5,0.5]);
    xlim([-1,1]);
    ylim([-1,1]);
    drawnow;
    pause(0.001);
end
    



