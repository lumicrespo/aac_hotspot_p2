// reset the current Gem5 statistics, i.e., it sets all counters to zero.
#define M5resetstats() \
 asm volatile( \
 "reset_stats:\n\t" \
 "mov x0, #0; mov x1, #0; .inst 0XFF000110 | (0x40 << 16);" :: \
 : "x0", "x1")
// dump the current Gem5 statistics, thus introducing a new section on stats.txt
// Each statistics section is delimited by the following lines
// ---------- Begin Simulation Statistics ----------
// your statistics here
// ---------- End Simulation Statistics ----------
#define M5dumpstats() \
 asm volatile( \
 "dump_stats:\n\t" \
 "mov x0, #0; mov x1, #0; .inst 0XFF000110 | (0x41 << 16);" :: \
 : "x0", "x1")
// dump the current Gem5 statistics, and reset counters to zero
#define M5resetdumpstats() \
 asm volatile( \
 "reset_and_dump_stats:\n\t" \
 "mov x0, #0; mov x1, #0; .inst 0XFF000110 | (0x42 << 16);" :: \
 : "x0", "x1")
