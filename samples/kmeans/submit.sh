unset DISPLAY
aoc -board=a10gx -time time.out -time-passes -regtest_mode -v -fpc -fp-relaxed --opt-arg -nocaching -regtest_mode -report -I $INTELFPGAOCLSDKROOT/include/kernel_headers kmeans_aocl.cl

