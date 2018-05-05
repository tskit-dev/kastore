project('kastore', 'c')
add_global_arguments([
    '-std=c99', '-Wall', '-Wextra', '-Werror', '-pedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common'], language : 'c')
link_args =  ['-Wl,--wrap=malloc']
cunit_dep = dependency('cunit')
executable('tests', ['tests.c', 'kastore.c'], dependencies: cunit_dep, link_args:link_args)
