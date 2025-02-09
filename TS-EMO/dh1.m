function y = dh1(x)
  dim = size(x, 2);
  insert(py.sys.path, int32(0), '/storage/home/kkc5441/work/mobo/TS-EMO/BC_function.py');

  dh1 = py.importlib.import_module('dh1_func');
  py_x = py.numpy.array(reshape(x, [], dim));
  py_y = dh1.dh1_eval(py_x, dim);
  y = double(py.array.array('d', py.numpy.nditer(py_y)));

end
