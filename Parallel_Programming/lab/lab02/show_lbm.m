%% set mesh size here
NX=256;
NY=256;

filename = 'output_velocity.txt';
delimiter = ' ';
formatSpec = '%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
outputvelocity = [dataArray{1:end-1}];

ux = outputvelocity(:,2);
uy = outputvelocity(:,3);
SOLID = outputvelocity(:,1)==1;
ux(SOLID)=0; uy(SOLID)=0;ux=reshape(ux,NX,NY)';uy=reshape(uy,NX,NY)';
figure(1);clf;hold on;colormap(gray(2));image(2-reshape(SOLID,NX,NY)');
quiver(1:NX,1:NY,ux,uy,1.5,'b');axis([0.5 NX+0.5 0.5 NY+0.5]);axis image;
% title(['Velocity field after ',num2str(t_),' time steps']);