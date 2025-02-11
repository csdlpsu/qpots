function y = branincurrin(x)
    % Ensure the Python script directory is in the Python search path
    insert(py.sys.path, int32(0), '/storage/home/kkc5441/work/mobo/TS-EMO/BC_function.py');
   
    BCModule = py.importlib.import_module('BC_function');
    
    % Prepare input as a 2D numpy array. Reshape x if necessary.
    py_x = py.numpy.array(reshape(x, [], 2));
    
    % Call the Python function
    py_y = BCModule.BC_evaluate(py_x);
    
    % Convert the Python output (numpy array) to a MATLAB array
    y = double(py.array.array('d', py.numpy.nditer(py_y)));
end
