project('PSI', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
  license: 'GPL-3.0-or-later',
  meson_version: '>=0.60.0',
  version: '2.0'
)

cxx = meson.get_compiler('cpp')
host_system = host_machine.system()
host_cpu = host_machine.cpu_family()
gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = dependency('', required: false)
  install_dir = get_option('libdir') / 'vapoursynth'
endif

# Use eigen from meson wrap (force fallback to always use wrap instead of system)
eigen_dep = dependency('', required: false, fallback: ['eigen', 'eigen_dep'])

blas_lapack_dep = []
compile_args = []
link_args = []

if host_system == 'darwin'

  link_args += ['-framework', 'Accelerate']
  compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']

else

  mkl_deps = []

  if host_cpu == 'x86' or host_cpu == 'x86_64'
    mkl_deps = [
      dependency('mkl-static-lp64-iomp', required: false),
      dependency('mkl-dynamic-lp64-iomp', required: false),
      dependency('mkl', required: false)
    ]
  endif

  armpl_dep = []
  if host_cpu.startswith('arm') and host_system != 'darwin'
    armpl_dep = [dependency('armpl', required: false)]
  endif

  openblas_dep = dependency('openblas', required: false)
  
  blas_dep = dependency('blas', required: false)
  lapacke_dep = dependency('lapacke', required: false)
  
  found_high_perf = false
  
  if mkl_deps.length() > 0 and (mkl_deps[0].found() or mkl_deps[1].found() or mkl_deps[2].found())
    foreach mkl : mkl_deps
      if mkl.found()
        blas_lapack_dep = [mkl]
        compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE', '-DMKL_LP64']
        found_high_perf = true
        break
      endif
    endforeach
  endif
  
  if not found_high_perf and armpl_dep.length() > 0 and armpl_dep[0].found()
    blas_lapack_dep = armpl_dep
    compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']
    found_high_perf = true
  endif
  
  if not found_high_perf and openblas_dep.found()
    blas_lapack_dep = [openblas_dep]
    compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']
    found_high_perf = true
  endif
  
  if not found_high_perf
    if blas_dep.found()
      blas_lapack_dep += [blas_dep]
      compile_args += ['-DEIGEN_USE_BLAS']
    endif
    if lapacke_dep.found()
      blas_lapack_dep += [lapacke_dep]
      compile_args += ['-DEIGEN_USE_LAPACKE']
    endif
  endif
endif


if compile_args.length() > 0
  add_project_arguments(compile_args, language: 'cpp')
endif


sources = [
  'PSI/PSI.cpp'
]

shared_module('PSI', sources,
  dependencies: [vapoursynth_dep, eigen_dep] + blas_lapack_dep,
  link_args: link_args,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)