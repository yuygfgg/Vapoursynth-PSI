project('eigen', 'cpp')

eigen_inc = include_directories('.')
eigen_dep = declare_dependency(include_directories: eigen_inc)

meson.override_dependency('eigen3', eigen_dep) 