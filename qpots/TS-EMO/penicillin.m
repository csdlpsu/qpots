function y = penicillin(x)
  dim = size(x,2); 
  module = py.importlib.import_module('penicillin_func');

  py_x = py.numpy.array(reshape(x, [], dim));
  py_y = module.Penicillin_evaluate(x);
  y = double(py.array.array('d', py.numpy.nditer(py_y)));
end
