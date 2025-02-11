function y = dtlz1(x)
  dim = size(x, 2);
  module = py.importlib.import_module('dtlz1_func');

  py_x = py.numpy.array(reshape(x, [], dim));
  py_y = module.dtlz1(py_x, dim);

  y = double(py.array.array('d', py.numpy.nditer(py_y)));
end
