project('kastore', 'c')
add_global_arguments([
    '-std=c99', '-Wall', '-Wextra', '-Werror', '-pedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common'], language : 'c')
cunit_dep = dependency('cunit')
executable('tests', ['tests.c', 'kastore.c'], dependencies: cunit_dep)

link_args =  ['-Wl,--wrap=malloc', '-Wl,--wrap=realloc', '-Wl,--wrap=calloc']
executable('malloc_tests', ['malloc_tests.c', 'kastore.c'], dependencies: cunit_dep, link_args:link_args)
