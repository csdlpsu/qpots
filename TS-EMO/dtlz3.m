function y = dtlz3(x)
  dim = size(x,2);
  module = py.importlib.import_module('dtlz3');

  py_x = py.numpy.array(reshape(x, [], dim));
  py_y = module.dtlz3(py_x, dim);

  y = double(py.array.array('d', py.numpy.nditer(py_y)));
end

  
