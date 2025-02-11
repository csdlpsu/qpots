function y = vehicle(x)
    dim = size(x, 2);
    module = py.importlib.import_module('vehiclesafety');
    py_x = py.numpy.array(reshape(x, [], dim));
    py_y = module.vehiclesafety(py_x);
    y = double(py.array.array('d', py.numpy.nditer(py_y)));
end
