project('kastore', ['c', 'cpp'], default_options: ['c_std=c99', 'cpp_std=c++11'])

if not meson.is_subproject()
    add_global_arguments([
        '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
        '-Wmissing-prototypes',  '-Wstrict-prototypes',
        '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
        '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
        '-fshort-enums', '-fno-common'], language : 'c')
endif

# Subprojects should compile in the static library for simplicity.
inc = include_directories('.')
kastore = static_library('kastore', 'kastore.c')
kastore_dep = declare_dependency(link_with : kastore, include_directories: inc)

if not meson.is_subproject()

    # The shared library can be installed into the system.
    install_headers('kastore.h')
    shared_library('kastore', 'kastore.c', install: true)
    executable('example', ['example.c'], link_with: kastore)

    cunit_dep = dependency('cunit')
    executable('tests', ['tests.c', 'kastore.c'], dependencies: cunit_dep)

    executable('cpp_tests', ['cpp_tests.cpp'], link_with: kastore)

    executable('malloc_tests', ['malloc_tests.c', 'kastore.c'], 
        dependencies: cunit_dep, 
        link_args:['-Wl,--wrap=malloc', '-Wl,--wrap=realloc', '-Wl,--wrap=calloc'])

    executable('io_tests', ['io_tests.c', 'kastore.c'], 
        dependencies: cunit_dep, 
        link_args:[
            '-Wl,--wrap=fwrite', 
            '-Wl,--wrap=fread', 
            '-Wl,--wrap=fclose',
            '-Wl,--wrap=fseek'])
endif
