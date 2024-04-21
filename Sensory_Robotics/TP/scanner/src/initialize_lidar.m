function lidar = initialize_lidar ()

    % #####
    % step1: initialization of MATLAB serial object
    lidar=serial('COM8', 'BaudRate', 19200);
    set(lidar, 'Timeout', 1);
    set(lidar, 'InputBufferSize', 40000);
    set(lidar, 'Terminator', 'LF');
    
    
    
    
    % #####
    % step2: ask the driver to use the newer communication protocol + setting
    % up higher speed on the serial line
    fopen(lidar);
    pause(0.1);
    fprintf(lidar, 'SCIP2.0');
    pause(0.1);
    read_counter = 12;
    [data, num_of_bytes] = fscanf(lidar);
    read_counter = read_counter-num_of_bytes;
    while read_counter>0
        [data, num_of_bytes] = fscanf(lidar);
        read_counter = read_counter-num_of_bytes;
    end
    
    fprintf(lidar, 'SS115200');
    pause(0.1);
    read_counter = 15;
    [data, num_of_bytes] = fscanf(lidar);
    read_counter = read_counter-num_of_bytes;
    while read_counter>0
        [data, num_of_bytes] = fscanf(lidar);
        read_counter = read_counter-num_of_bytes;
    end
    set(lidar, 'BaudRate', 115200);
    
    
    
    
    % #####
    % step3: gathering some state-information from the laser-scanner.
    % Later on you can comment out this step, this has here just demonstration
    % purposes right now.
    fprintf(lidar, 'II');
    pause(0.1);
    [data, num_of_bytes] = fscanf(lidar);
    read_data = [data];
    while num_of_bytes>0
        [data, num_of_bytes] = fscanf(lidar);
        read_data = [read_data, data];
    end
    read_data
end