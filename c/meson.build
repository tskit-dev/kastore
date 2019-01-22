project('kastore', ['c', 'cpp'])
add_global_arguments([
    '-std=c99', '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common'], language : 'c')

shared_library('kastore', 'kastore.c')
static_lib = static_library('kastore', 'kastore.c')

cunit_dep = dependency('cunit')
executable('tests', ['tests.c', 'kastore.c'], dependencies: cunit_dep)

executable('cpp_tests', ['cpp_tests.cpp'], link_with: static_lib)

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

