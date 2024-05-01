
lidar = initialize_lidar();
tilt_angles = phidget_accelerometer_test();
fprintf(lidar,'MD0044072501101');
pause(0.1);
measurement_data = read_lidar_data(lidar);
fprintf(lidar,'QT');
fclose(lidar);

save ('measurement_data_case18.mat', "tilt_angles","measurement_data");