c KSU TRIGA Mark II (plugs are all air)
c
c Core with no 12% elements
c
c CELLS:
    1     0         1 :-20 :39  $ OUTSIDE
    2     6      -1 (((-1 2 -38 ):(-2 7 -38 44 ): $ POOL ELEMENTS
            (-17 18 -38 30 ):(-18 20 -38 ):(-7 17 -38 37 )):
            (-15 18 -38 43 ))#7 #14
    3     6      -1 -34 33 -7 11  $ RSR SPACE/VOLUME
    4     7  -2.699 -35 32 -9 12 #3  $ RSR WALLS
    5     2    -1.6 -36 31 -9 16 
            #3 #4 #7 #14             $ REFLECTOR GRAPHITE
    6     7  -2.699 -37 30 -7 17 #3 #4 #5 #7 #14  $ CANNISTER/WALLS
    7     7  -2.699 31 -40 90 -114 26  $ NEBP AL CASE
    9     0         (-30 -7 15 ):(-44 -2 7 ) fill=7 $ CORE SPACE
   10     7  -2.699 -15 18 -43 203  $ Lower grid plate
   11     0         -15 18 -203  fill=4 $ CT penetration, lower grid plate
   13    15  -2.4 -1 20 38 -39 #7 #14 #20 #21 #22 #23    $ concrete outer
   14     3  -0.000124 31 -114 26 -90          $ PLUG 0 
c    15     3  -0.000124       111 -114 26   -90 $ PLUG 1    
c    16     3  -0.000124       112 -113 26   -90 $ PLUG 2
c    17     3  -0.000124       113 -114 26   -90 $ PLUG 3
c    18     3  -0.000124       114 -38  26   -90 $ PLUG 4
   20     7  -2.699   114 -41 91 -39 26  $ NEBP AL CASE outer
   21     5  -1.0000  114 -91 46 -39 26  $ NEBP AL CASE outer BP collimator
   22     3  -0.000124 114 -115 -46 26   $ Borated Poly Collimator
   23     3  -0.000124 115 -39 -46 26   $ NEBP Exit Detector
c ** UNIVERSES
c ******************************************************************************
c ** WATER IN CORE AREA: UNIVERSE = [8]
   95     6   -1         20              u=8  $ Water above pool bottom
   96     7   -2.699    -20              u=8  $ Stuff below pool bottom
c ** CONTROL ROD UNIVERSES
c ** NOTE: Holes in guide tubes modeled as reduced Al density
c    (101, 110, 120, 126)
c ** PULSE ROD: UNIVERSE = [21]
  101     7  -2.12  210                  u=21 $ Guide Tube
  102     7  -2.699 50 -68 213 -214      u=21 $ Rod cladding
  103    11  -2.0   50 -68 -213          u=21 $ Pulse rod
  104     9  -7.9   68 -217 218          u=21 $ SS extension
  105     6  -1    -210 #102 #103 #104   u=21 $ Water
c ** SAFETY ROD: UNIVERSE = [22]
  110     7  -2.12  210                  u=22 $ Guide Tube
  111    13  -2.52  51 -68 -212          u=22 $ Safety rod
  112     9  -7.9   68 -217 218          u=22 $ SS extension
  113     6  -1    -210 #111 #112        u=22 $ Water
c ** SHIM ROD: UNIVERSE = [23]
  120     7  -2.12  210                  u=23 $ Guide Tube
  121     7  -2.699 52 -68 211 -212      u=23 $ Rod cladding
  122    12  -2.48  52 -68 -211          u=23 $ Shim/safety rod
  123     9  -7.9   68 -217 218          u=23 $ SS extension
  124     6  -1   -210 #121 #122 #123    u=23 $ Water
c ** REGULATING ROD: UNIVERSE = [24]
  126     7  -2.12 219                   u=24 $ Guide Tube
  127     7  -2.699 53 -68  215 -216     u=24 $ Rod cladding
  128    12  -2.48  53 -68 -215          u=24 $ Regulating rod
  129     9  -7.9   68 218 -217          u=24 $ SS extension
  130     6  -1    -219 #127 #128 #129   u=24 $ Water
c ******************************************************************************
c ** FUEL UNIVERSES: Individual FE's modeled for burnup as of 3-31-10
c ==============================================================================
  201000   6   -1 -21 201                  u=201   $ Water around lower pin
  201001   9   -7.9 -21 -201               u=201   $ Bottom pin
  201002   9   -7.9 -14 21                 u=201   $ BOTTOM SS cap
  201003   2   -1.6 14 -13 -202            u=201   $ BOTTOM Axial reflector
  201004   9   -7.9 14 -4 202              u=201   $ Cladding
  201005   8   -6.5 13 -10 -200            u=201   $ Zirc FILLER
  201006   2   -1.6 10 -6 -202             u=201   $ TOP Axial reflector
  201007   3   -0.001239 6 -4 -202         u=201   $ Air gap
  201008   9   -7.9 4 -3                   u=201   $ TOP SS cap
  201009   9   -7.9 3 -201                 u=201   $ Upper pin
  201010   6   -1 3 201                    u=201   $ Water around pin
  201100  2011  -6.240 100 -10  200  -202  u=201   $ ELEMENT B-1.1-1  S/N:6315
  201110  2012  -6.239 101 -100 200  -202  u=201   $ ELEMENT B-1.2-1  S/N:6315
  201120  2013  -6.239 102 -101 200  -202  u=201   $ ELEMENT B-1.3-1  S/N:6315
  201130  2014  -6.238 103 -102 200  -202  u=201   $ ELEMENT B-1.4-1  S/N:6315
  201140  2013  -6.239 104 -103 200  -202  u=201   $ ELEMENT B-1.5-1  S/N:6315
  201150  2012  -6.239 105 -104 200  -202  u=201   $ ELEMENT B-1.6-1  S/N:6315
  201160  2011  -6.240 13  -105 200  -202  u=201   $ ELEMENT B-1.7-1  S/N:6315
c ==============================================================================
  202000 like 201000 but u=202
  202001 like 201001 but u=202
  202002 like 201002 but u=202
  202003 like 201003 but u=202
  202004 like 201004 but u=202
  202005 like 201005 but u=202
  202006 like 201006 but u=202
  202007 like 201007 but u=202
  202008 like 201008 but u=202
  202009 like 201009 but u=202
  202010 like 201010 but u=202
  202100 like 201100 but mat=2021 rho=-6.386 u=202  $ ELEMENT B-2.1-1  S/N:10880
  202110 like 201110 but mat=2022 rho=-6.386 u=202  $ ELEMENT B-2.2-1  S/N:10880
  202120 like 201120 but mat=2023 rho=-6.386 u=202  $ ELEMENT B-2.3-1  S/N:10880
  202130 like 201130 but mat=2024 rho=-6.386 u=202  $ ELEMENT B-2.4-1  S/N:10880
  202140 like 201140 but mat=2023 rho=-6.386 u=202  $ ELEMENT B-2.5-1  S/N:10880
  202150 like 201150 but mat=2022 rho=-6.386 u=202  $ ELEMENT B-2.6-1  S/N:10880
  202160 like 201160 but mat=2021 rho=-6.386 u=202  $ ELEMENT B-2.7-1  S/N:10880
c ==============================================================================
  203000 like 201000 but u=203
  203001 like 201001 but u=203
  203002 like 201002 but u=203
  203003 like 201003 but u=203
  203004 like 201004 but u=203
  203005 like 201005 but u=203
  203006 like 201006 but u=203
  203007 like 201007 but u=203
  203008 like 201008 but u=203
  203009 like 201009 but u=203
  203010 like 201010 but u=203
  203100 like 201100 but mat=2031 rho=-6.240 u=203  $ ELEMENT B-3.1-1  S/N:6577
  203110 like 201110 but mat=2032 rho=-6.239 u=203  $ ELEMENT B-3.2-1  S/N:6577
  203120 like 201120 but mat=2033 rho=-6.239 u=203  $ ELEMENT B-3.3-1  S/N:6577
  203130 like 201130 but mat=2034 rho=-6.238 u=203  $ ELEMENT B-3.4-1  S/N:6577
  203140 like 201140 but mat=2033 rho=-6.239 u=203  $ ELEMENT B-3.5-1  S/N:6577
  203150 like 201150 but mat=2032 rho=-6.239 u=203  $ ELEMENT B-3.6-1  S/N:6577
  203160 like 201160 but mat=2031 rho=-6.240 u=203  $ ELEMENT B-3.7-1  S/N:6577
c ==============================================================================
  204000 like 201000 but u=204
  204001 like 201001 but u=204
  204002 like 201002 but u=204
  204003 like 201003 but u=204
  204004 like 201004 but u=204
  204005 like 201005 but u=204
  204006 like 201006 but u=204
  204007 like 201007 but u=204
  204008 like 201008 but u=204
  204009 like 201009 but u=204
  204010 like 201010 but u=204
  204100 like 201100 but mat=2041 rho=-6.306 u=204  $ ELEMENT B-4.1-1  S/N:2966
  204110 like 201110 but mat=2042 rho=-6.305 u=204  $ ELEMENT B-4.2-1  S/N:2966
  204120 like 201120 but mat=2043 rho=-6.304 u=204  $ ELEMENT B-4.3-1  S/N:2966
  204130 like 201130 but mat=2044 rho=-6.304 u=204  $ ELEMENT B-4.4-1  S/N:2966
  204140 like 201140 but mat=2043 rho=-6.304 u=204  $ ELEMENT B-4.5-1  S/N:2966
  204150 like 201150 but mat=2042 rho=-6.305 u=204  $ ELEMENT B-4.6-1  S/N:2966
  204160 like 201160 but mat=2041 rho=-6.306 u=204  $ ELEMENT B-4.7-1  S/N:2966
c ==============================================================================
  205000 like 201000 but u=205
  205001 like 201001 but u=205
  205002 like 201002 but u=205
  205003 like 201003 but u=205
  205004 like 201004 but u=205
  205005 like 201005 but u=205
  205006 like 201006 but u=205
  205007 like 201007 but u=205
  205008 like 201008 but u=205
  205009 like 201009 but u=205
  205010 like 201010 but u=205
  205100 like 201100 but mat=2051 rho=-6.406 u=205  $ ELEMENT B-5.1-1  S/N:10895
  205110 like 201110 but mat=2052 rho=-6.406 u=205  $ ELEMENT B-5.2-1  S/N:10895
  205120 like 201120 but mat=2053 rho=-6.406 u=205  $ ELEMENT B-5.3-1  S/N:10895
  205130 like 201130 but mat=2054 rho=-6.406 u=205  $ ELEMENT B-5.4-1  S/N:10895
  205140 like 201140 but mat=2053 rho=-6.406 u=205  $ ELEMENT B-5.5-1  S/N:10895
  205150 like 201150 but mat=2052 rho=-6.406 u=205  $ ELEMENT B-5.6-1  S/N:10895
  205160 like 201160 but mat=2051 rho=-6.406 u=205  $ ELEMENT B-5.7-1  S/N:10895
c ==============================================================================
  206000 like 201000 but u=206
  206001 like 201001 but u=206
  206002 like 201002 but u=206
  206003 like 201003 but u=206
  206004 like 201004 but u=206
  206005 like 201005 but u=206
  206006 like 201006 but u=206
  206007 like 201007 but u=206
  206008 like 201008 but u=206
  206009 like 201009 but u=206
  206010 like 201010 but u=206
  206100 like 201100 but mat=2061 rho=-6.372 u=206  $ ELEMENT B-6.1-1  S/N:11341
  206110 like 201110 but mat=2062 rho=-6.372 u=206  $ ELEMENT B-6.2-1  S/N:11341
  206120 like 201120 but mat=2063 rho=-6.372 u=206  $ ELEMENT B-6.3-1  S/N:11341
  206130 like 201130 but mat=2064 rho=-6.372 u=206  $ ELEMENT B-6.4-1  S/N:11341
  206140 like 201140 but mat=2063 rho=-6.372 u=206  $ ELEMENT B-6.5-1  S/N:11341
  206150 like 201150 but mat=2062 rho=-6.372 u=206  $ ELEMENT B-6.6-1  S/N:11341
  206160 like 201160 but mat=2061 rho=-6.372 u=206  $ ELEMENT B-6.7-1  S/N:11341
c ==============================================================================
  301000 like 201000 but u=301
  301001 like 201001 but u=301
  301002 like 201002 but u=301
  301003 like 201003 but u=301
  301004 like 201004 but u=301
  301005 like 201005 but u=301
  301006 like 201006 but u=301
  301007 like 201007 but u=301
  301008 like 201008 but u=301
  301009 like 201009 but u=301
  301010 like 201010 but u=301
  301100 like 201100 but mat=3011 rho=-6.406 u=301  $ ELEMENT C-1.1-1  S/N:11351
  301110 like 201110 but mat=3012 rho=-6.406 u=301  $ ELEMENT C-1.2-1  S/N:11351
  301120 like 201120 but mat=3013 rho=-6.406 u=301  $ ELEMENT C-1.3-1  S/N:11351
  301130 like 201130 but mat=3014 rho=-6.406 u=301  $ ELEMENT C-1.4-1  S/N:11351
  301140 like 201140 but mat=3013 rho=-6.406 u=301  $ ELEMENT C-1.5-1  S/N:11351
  301150 like 201150 but mat=3012 rho=-6.406 u=301  $ ELEMENT C-1.6-1  S/N:11351
  301160 like 201160 but mat=3011 rho=-6.406 u=301  $ ELEMENT C-1.7-1  S/N:11351
c ==============================================================================
  302000 like 201000 but u=302
  302001 like 201001 but u=302
  302002 like 201002 but u=302
  302003 like 201003 but u=302
  302004 like 201004 but u=302
  302005 like 201005 but u=302
  302006 like 201006 but u=302
  302007 like 201007 but u=302
  302008 like 201008 but u=302
  302009 like 201009 but u=302
  302010 like 201010 but u=302
  302100 like 201100 but mat=3021 rho=-6.240 u=302  $ ELEMENT C-2.1-1  S/N:6316
  302110 like 201110 but mat=3022 rho=-6.239 u=302  $ ELEMENT C-2.2-1  S/N:6316
  302120 like 201120 but mat=3023 rho=-6.239 u=302  $ ELEMENT C-2.3-1  S/N:6316
  302130 like 201130 but mat=3024 rho=-6.239 u=302  $ ELEMENT C-2.4-1  S/N:6316
  302140 like 201140 but mat=3023 rho=-6.239 u=302  $ ELEMENT C-2.5-1  S/N:6316
  302150 like 201150 but mat=3022 rho=-6.239 u=302  $ ELEMENT C-2.6-1  S/N:6316
  302160 like 201160 but mat=3021 rho=-6.240 u=302  $ ELEMENT C-2.7-1  S/N:6316
c ==============================================================================
  303000 like 201000 but u=303
  303001 like 201001 but u=303
  303002 like 201002 but u=303
  303003 like 201003 but u=303
  303004 like 201004 but u=303
  303005 like 201005 but u=303
  303006 like 201006 but u=303
  303007 like 201007 but u=303
  303008 like 201008 but u=303
  303009 like 201009 but u=303
  303010 like 201010 but u=303
  303100 like 201100 but mat=3031 rho=-6.339 u=303  $ ELEMENT C-3.1-1  S/N:2963
  303110 like 201110 but mat=3032 rho=-6.338 u=303  $ ELEMENT C-3.2-1  S/N:2963
  303120 like 201120 but mat=3033 rho=-6.338 u=303  $ ELEMENT C-3.3-1  S/N:2963
  303130 like 201130 but mat=3034 rho=-6.337 u=303  $ ELEMENT C-3.4-1  S/N:2963
  303140 like 201140 but mat=3033 rho=-6.338 u=303  $ ELEMENT C-3.5-1  S/N:2963
  303150 like 201150 but mat=3032 rho=-6.338 u=303  $ ELEMENT C-3.6-1  S/N:2963
  303160 like 201160 but mat=3031 rho=-6.339 u=303  $ ELEMENT C-3.7-1  S/N:2963
c ==============================================================================
  304000 like 201000 but u=304
  304001 like 201001 but u=304
  304002 like 201002 but u=304
  304003 like 201003 but u=304
  304004 like 201004 but u=304
  304005 like 201005 but u=304
  304006 like 201006 but u=304
  304007 like 201007 but u=304
  304008 like 201008 but u=304
  304009 like 201009 but u=304
  304010 like 201010 but u=304
  304100 like 201100 but mat=3041 rho=-6.407 u=304  $ ELEMENT C-4.1-1  S/N:10894
  304110 like 201110 but mat=3042 rho=-6.407 u=304  $ ELEMENT C-4.2-1  S/N:10894
  304120 like 201120 but mat=3043 rho=-6.406 u=304  $ ELEMENT C-4.3-1  S/N:10894
  304130 like 201130 but mat=3044 rho=-6.406 u=304  $ ELEMENT C-4.4-1  S/N:10894
  304140 like 201140 but mat=3043 rho=-6.406 u=304  $ ELEMENT C-4.5-1  S/N:10894
  304150 like 201150 but mat=3042 rho=-6.407 u=304  $ ELEMENT C-4.6-1  S/N:10894
  304160 like 201160 but mat=3041 rho=-6.407 u=304  $ ELEMENT C-4.7-1  S/N:10894
c ==============================================================================
  305000 like 201000 but u=305
  305001 like 201001 but u=305
  305002 like 201002 but u=305
  305003 like 201003 but u=305
  305004 like 201004 but u=305
  305005 like 201005 but u=305
  305006 like 201006 but u=305
  305007 like 201007 but u=305
  305008 like 201008 but u=305
  305009 like 201009 but u=305
  305010 like 201010 but u=305
  305100 like 201100 but mat=3051 rho=-6.588 u=305  $ ELEMENT C-5.1-1  S/N:2953
  305110 like 201110 but mat=3052 rho=-6.588 u=305  $ ELEMENT C-5.2-1  S/N:2953
  305120 like 201120 but mat=3053 rho=-6.587 u=305  $ ELEMENT C-5.3-1  S/N:2953
  305130 like 201130 but mat=3054 rho=-6.587 u=305  $ ELEMENT C-5.4-1  S/N:2953
  305140 like 201140 but mat=3053 rho=-6.587 u=305  $ ELEMENT C-5.5-1  S/N:2953
  305150 like 201150 but mat=3052 rho=-6.588 u=305  $ ELEMENT C-5.6-1  S/N:2953
  305160 like 201160 but mat=3051 rho=-6.588 u=305  $ ELEMENT C-5.7-1  S/N:2953
c ==============================================================================
  306000 like 201000 but u=306
  306001 like 201001 but u=306
  306002 like 201002 but u=306
  306003 like 201003 but u=306
  306004 like 201004 but u=306
  306005 like 201005 but u=306
  306006 like 201006 but u=306
  306007 like 201007 but u=306
  306008 like 201008 but u=306
  306009 like 201009 but u=306
  306010 like 201010 but u=306
  306100 like 201100 but mat=3061 rho=-6.074 u=306  $ ELEMENT C-6.1-1  S/N:3082
  306110 like 201110 but mat=3062 rho=-6.073 u=306  $ ELEMENT C-6.2-1  S/N:3082
  306120 like 201120 but mat=3063 rho=-6.071 u=306  $ ELEMENT C-6.3-1  S/N:3082
  306130 like 201130 but mat=3064 rho=-6.071 u=306  $ ELEMENT C-6.4-1  S/N:3082
  306140 like 201140 but mat=3063 rho=-6.071 u=306  $ ELEMENT C-6.5-1  S/N:3082
  306150 like 201150 but mat=3062 rho=-6.073 u=306  $ ELEMENT C-6.6-1  S/N:3082
  306160 like 201160 but mat=3061 rho=-6.074 u=306  $ ELEMENT C-6.7-1  S/N:3082
c ==============================================================================
  308000 like 201000 but u=308
  308001 like 201001 but u=308
  308002 like 201002 but u=308
  308003 like 201003 but u=308
  308004 like 201004 but u=308
  308005 like 201005 but u=308
  308006 like 201006 but u=308
  308007 like 201007 but u=308
  308008 like 201008 but u=308
  308009 like 201009 but u=308
  308010 like 201010 but u=308
  308100 like 201100 but mat=3081 rho=-6.331 u=308  $ ELEMENT C-8.1-1  S/N:2933
  308110 like 201110 but mat=3082 rho=-6.331 u=308  $ ELEMENT C-8.2-1  S/N:2933
  308120 like 201120 but mat=3083 rho=-6.330 u=308  $ ELEMENT C-8.3-1  S/N:2933
  308130 like 201130 but mat=3084 rho=-6.330 u=308  $ ELEMENT C-8.4-1  S/N:2933
  308140 like 201140 but mat=3083 rho=-6.330 u=308  $ ELEMENT C-8.5-1  S/N:2933
  308150 like 201150 but mat=3082 rho=-6.331 u=308  $ ELEMENT C-8.6-1  S/N:2933
  308160 like 201160 but mat=3081 rho=-6.331 u=308  $ ELEMENT C-8.7-1  S/N:2933
c ==============================================================================
  309000 like 201000 but u=309
  309001 like 201001 but u=309
  309002 like 201002 but u=309
  309003 like 201003 but u=309
  309004 like 201004 but u=309
  309005 like 201005 but u=309
  309006 like 201006 but u=309
  309007 like 201007 but u=309
  309008 like 201008 but u=309
  309009 like 201009 but u=309
  309010 like 201010 but u=309
  309100 like 201100 but mat=3091 rho=-6.240 u=309  $ ELEMENT C-9.1-1  S/N:6314
  309110 like 201110 but mat=3092 rho=-6.239 u=309  $ ELEMENT C-9.2-1  S/N:6314
  309120 like 201120 but mat=3093 rho=-6.239 u=309  $ ELEMENT C-9.3-1  S/N:6314
  309130 like 201130 but mat=3094 rho=-6.239 u=309  $ ELEMENT C-9.4-1  S/N:6314
  309140 like 201140 but mat=3093 rho=-6.239 u=309  $ ELEMENT C-9.5-1  S/N:6314
  309150 like 201150 but mat=3092 rho=-6.239 u=309  $ ELEMENT C-9.6-1  S/N:6314
  309160 like 201160 but mat=3091 rho=-6.240 u=309  $ ELEMENT C-9.7-1  S/N:6314
c ==============================================================================
  310000 like 201000 but u=310
  310001 like 201001 but u=310
  310002 like 201002 but u=310
  310003 like 201003 but u=310
  310004 like 201004 but u=310
  310005 like 201005 but u=310
  310006 like 201006 but u=310
  310007 like 201007 but u=310
  310008 like 201008 but u=310
  310009 like 201009 but u=310
  310010 like 201010 but u=310
  310100 like 201100 but mat=3101 rho=-6.240 u=310  $ ELEMENT C-10.1-1  S/N:6527
  310110 like 201110 but mat=3102 rho=-6.239 u=310  $ ELEMENT C-10.2-1  S/N:6527
  310120 like 201120 but mat=3103 rho=-6.239 u=310  $ ELEMENT C-10.3-1  S/N:6527
  310130 like 201130 but mat=3104 rho=-6.239 u=310  $ ELEMENT C-10.4-1  S/N:6527
  310140 like 201140 but mat=3103 rho=-6.239 u=310  $ ELEMENT C-10.5-1  S/N:6527
  310150 like 201150 but mat=3102 rho=-6.239 u=310  $ ELEMENT C-10.6-1  S/N:6527
  310160 like 201160 but mat=3101 rho=-6.240 u=310  $ ELEMENT C-10.7-1  S/N:6527
c ==============================================================================
  311000 like 201000 but u=311
  311001 like 201001 but u=311
  311002 like 201002 but u=311
  311003 like 201003 but u=311
  311004 like 201004 but u=311
  311005 like 201005 but u=311
  311006 like 201006 but u=311
  311007 like 201007 but u=311
  311008 like 201008 but u=311
  311009 like 201009 but u=311
  311010 like 201010 but u=311
  311100 like 201100 but mat=3111 rho=-6.240 u=311  $ ELEMENT C-11.1-1  S/N:6317
  311110 like 201110 but mat=3112 rho=-6.239 u=311  $ ELEMENT C-11.2-1  S/N:6317
  311120 like 201120 but mat=3113 rho=-6.239 u=311  $ ELEMENT C-11.3-1  S/N:6317
  311130 like 201130 but mat=3114 rho=-6.238 u=311  $ ELEMENT C-11.4-1  S/N:6317
  311140 like 201140 but mat=3113 rho=-6.239 u=311  $ ELEMENT C-11.5-1  S/N:6317
  311150 like 201150 but mat=3112 rho=-6.239 u=311  $ ELEMENT C-11.6-1  S/N:6317
  311160 like 201160 but mat=3111 rho=-6.240 u=311  $ ELEMENT C-11.7-1  S/N:6317
c ==============================================================================
  312000 like 201000 but u=312
  312001 like 201001 but u=312
  312002 like 201002 but u=312
  312003 like 201003 but u=312
  312004 like 201004 but u=312
  312005 like 201005 but u=312
  312006 like 201006 but u=312
  312007 like 201007 but u=312
  312008 like 201008 but u=312
  312009 like 201009 but u=312
  312010 like 201010 but u=312
  312100 like 201100 but mat=3121 rho=-6.240 u=312  $ ELEMENT C-12.1-1  S/N:6526
  312110 like 201110 but mat=3122 rho=-6.240 u=312  $ ELEMENT C-12.2-1  S/N:6526
  312120 like 201120 but mat=3123 rho=-6.239 u=312  $ ELEMENT C-12.3-1  S/N:6526
  312130 like 201130 but mat=3124 rho=-6.239 u=312  $ ELEMENT C-12.4-1  S/N:6526
  312140 like 201140 but mat=3123 rho=-6.239 u=312  $ ELEMENT C-12.5-1  S/N:6526
  312150 like 201150 but mat=3122 rho=-6.240 u=312  $ ELEMENT C-12.6-1  S/N:6526
  312160 like 201160 but mat=3121 rho=-6.240 u=312  $ ELEMENT C-12.7-1  S/N:6526
c ==============================================================================
  401000 like 201000 but u=401
  401001 like 201001 but u=401
  401002 like 201002 but u=401
  401003 like 201003 but u=401
  401004 like 201004 but u=401
  401005 like 201005 but u=401
  401006 like 201006 but u=401
  401007 like 201007 but u=401
  401008 like 201008 but u=401
  401009 like 201009 but u=401
  401010 like 201010 but u=401
  401100 like 201100 but mat=4011 rho=-6.073 u=401  $ ELEMENT D-1.1-1  S/N:3380
  401110 like 201110 but mat=4012 rho=-6.071 u=401  $ ELEMENT D-1.2-1  S/N:3380
  401120 like 201120 but mat=4013 rho=-6.070 u=401  $ ELEMENT D-1.3-1  S/N:3380
  401130 like 201130 but mat=4014 rho=-6.070 u=401  $ ELEMENT D-1.4-1  S/N:3380
  401140 like 201140 but mat=4013 rho=-6.070 u=401  $ ELEMENT D-1.5-1  S/N:3380
  401150 like 201150 but mat=4012 rho=-6.071 u=401  $ ELEMENT D-1.6-1  S/N:3380
  401160 like 201160 but mat=4011 rho=-6.073 u=401  $ ELEMENT D-1.7-1  S/N:3380
c ==============================================================================
  402000 like 201000 but u=402
  402001 like 201001 but u=402
  402002 like 201002 but u=402
  402003 like 201003 but u=402
  402004 like 201004 but u=402
  402005 like 201005 but u=402
  402006 like 201006 but u=402
  402007 like 201007 but u=402
  402008 like 201008 but u=402
  402009 like 201009 but u=402
  402010 like 201010 but u=402
  402100 like 201100 but mat=4021 rho=-6.237 u=402  $ ELEMENT D-2.1-1  S/N:3330
  402110 like 201110 but mat=4022 rho=-6.236 u=402  $ ELEMENT D-2.2-1  S/N:3330
  402120 like 201120 but mat=4023 rho=-6.235 u=402  $ ELEMENT D-2.3-1  S/N:3330
  402130 like 201130 but mat=4024 rho=-6.235 u=402  $ ELEMENT D-2.4-1  S/N:3330
  402140 like 201140 but mat=4023 rho=-6.235 u=402  $ ELEMENT D-2.5-1  S/N:3330
  402150 like 201150 but mat=4022 rho=-6.236 u=402  $ ELEMENT D-2.6-1  S/N:3330
  402160 like 201160 but mat=4021 rho=-6.237 u=402  $ ELEMENT D-2.7-1  S/N:3330
c ==============================================================================
  403000 like 201000 but u=403
  403001 like 201001 but u=403
  403002 like 201002 but u=403
  403003 like 201003 but u=403
  403004 like 201004 but u=403
  403005 like 201005 but u=403
  403006 like 201006 but u=403
  403007 like 201007 but u=403
  403008 like 201008 but u=403
  403009 like 201009 but u=403
  403010 like 201010 but u=403
  403100 like 201100 but mat=4031 rho=-6.242 u=403  $ ELEMENT D-3.1-1  S/N:10706
  403110 like 201110 but mat=4032 rho=-6.242 u=403  $ ELEMENT D-3.2-1  S/N:10706
  403120 like 201120 but mat=4033 rho=-6.242 u=403  $ ELEMENT D-3.3-1  S/N:10706
  403130 like 201130 but mat=4034 rho=-6.242 u=403  $ ELEMENT D-3.4-1  S/N:10706
  403140 like 201140 but mat=4033 rho=-6.242 u=403  $ ELEMENT D-3.5-1  S/N:10706
  403150 like 201150 but mat=4032 rho=-6.242 u=403  $ ELEMENT D-3.6-1  S/N:10706
  403160 like 201160 but mat=4031 rho=-6.242 u=403  $ ELEMENT D-3.7-1  S/N:10706
c ==============================================================================
  405000 like 201000 but u=405
  405001 like 201001 but u=405
  405002 like 201002 but u=405
  405003 like 201003 but u=405
  405004 like 201004 but u=405
  405005 like 201005 but u=405
  405006 like 201006 but u=405
  405007 like 201007 but u=405
  405008 like 201008 but u=405
  405009 like 201009 but u=405
  405010 like 201010 but u=405
  405100 like 201100 but mat=4051 rho=-6.072 u=405  $ ELEMENT D-5.1-1  S/N:3144
  405110 like 201110 but mat=4052 rho=-6.071 u=405  $ ELEMENT D-5.2-1  S/N:3144
  405120 like 201120 but mat=4053 rho=-6.070 u=405  $ ELEMENT D-5.3-1  S/N:3144
  405130 like 201130 but mat=4054 rho=-6.069 u=405  $ ELEMENT D-5.4-1  S/N:3144
  405140 like 201140 but mat=4053 rho=-6.070 u=405  $ ELEMENT D-5.5-1  S/N:3144
  405150 like 201150 but mat=4052 rho=-6.071 u=405  $ ELEMENT D-5.6-1  S/N:3144
  405160 like 201160 but mat=4051 rho=-6.072 u=405  $ ELEMENT D-5.7-1  S/N:3144
c ==============================================================================
  406000 like 201000 but u=406
  406001 like 201001 but u=406
  406002 like 201002 but u=406
  406003 like 201003 but u=406
  406004 like 201004 but u=406
  406005 like 201005 but u=406
  406006 like 201006 but u=406
  406007 like 201007 but u=406
  406008 like 201008 but u=406
  406009 like 201009 but u=406
  406010 like 201010 but u=406
  406100 like 201100 but mat=4061 rho=-6.404 u=406  $ ELEMENT D-6.1-1  S/N:6224
  406110 like 201110 but mat=4062 rho=-6.404 u=406  $ ELEMENT D-6.2-1  S/N:6224
  406120 like 201120 but mat=4063 rho=-6.403 u=406  $ ELEMENT D-6.3-1  S/N:6224
  406130 like 201130 but mat=4064 rho=-6.403 u=406  $ ELEMENT D-6.4-1  S/N:6224
  406140 like 201140 but mat=4063 rho=-6.403 u=406  $ ELEMENT D-6.5-1  S/N:6224
  406150 like 201150 but mat=4062 rho=-6.404 u=406  $ ELEMENT D-6.6-1  S/N:6224
  406160 like 201160 but mat=4061 rho=-6.404 u=406  $ ELEMENT D-6.7-1  S/N:6224
c ==============================================================================
  407000 like 201000 but u=407
  407001 like 201001 but u=407
  407002 like 201002 but u=407
  407003 like 201003 but u=407
  407004 like 201004 but u=407
  407005 like 201005 but u=407
  407006 like 201006 but u=407
  407007 like 201007 but u=407
  407008 like 201008 but u=407
  407009 like 201009 but u=407
  407010 like 201010 but u=407
  407100 like 201100 but mat=4071 rho=-6.237 u=407  $ ELEMENT D-7.1-1  S/N:3018
  407110 like 201110 but mat=4072 rho=-6.235 u=407  $ ELEMENT D-7.2-1  S/N:3018
  407120 like 201120 but mat=4073 rho=-6.234 u=407  $ ELEMENT D-7.3-1  S/N:3018
  407130 like 201130 but mat=4074 rho=-6.233 u=407  $ ELEMENT D-7.4-1  S/N:3018
  407140 like 201140 but mat=4073 rho=-6.234 u=407  $ ELEMENT D-7.5-1  S/N:3018
  407150 like 201150 but mat=4072 rho=-6.235 u=407  $ ELEMENT D-7.6-1  S/N:3018
  407160 like 201160 but mat=4071 rho=-6.237 u=407  $ ELEMENT D-7.7-1  S/N:3018
c ==============================================================================
  408000 like 201000 but u=408
  408001 like 201001 but u=408
  408002 like 201002 but u=408
  408003 like 201003 but u=408
  408004 like 201004 but u=408
  408005 like 201005 but u=408
  408006 like 201006 but u=408
  408007 like 201007 but u=408
  408008 like 201008 but u=408
  408009 like 201009 but u=408
  408010 like 201010 but u=408
  408100 like 201100 but mat=4081 rho=-6.073 u=408  $ ELEMENT D-8.1-1  S/N:3105
  408110 like 201110 but mat=4082 rho=-6.071 u=408  $ ELEMENT D-8.2-1  S/N:3105
  408120 like 201120 but mat=4083 rho=-6.070 u=408  $ ELEMENT D-8.3-1  S/N:3105
  408130 like 201130 but mat=4084 rho=-6.070 u=408  $ ELEMENT D-8.4-1  S/N:3105
  408140 like 201140 but mat=4083 rho=-6.070 u=408  $ ELEMENT D-8.5-1  S/N:3105
  408150 like 201150 but mat=4082 rho=-6.071 u=408  $ ELEMENT D-8.6-1  S/N:3105
  408160 like 201160 but mat=4081 rho=-6.073 u=408  $ ELEMENT D-8.7-1  S/N:3105
c ==============================================================================
  409000 like 201000 but u=409
  409001 like 201001 but u=409
  409002 like 201002 but u=409
  409003 like 201003 but u=409
  409004 like 201004 but u=409
  409005 like 201005 but u=409
  409006 like 201006 but u=409
  409007 like 201007 but u=409
  409008 like 201008 but u=409
  409009 like 201009 but u=409
  409010 like 201010 but u=409
  409100 like 201100 but mat=4091 rho=-6.073 u=409  $ ELEMENT D-9.1-1  S/N:2452
  409110 like 201110 but mat=4092 rho=-6.072 u=409  $ ELEMENT D-9.2-1  S/N:2452
  409120 like 201120 but mat=4093 rho=-6.071 u=409  $ ELEMENT D-9.3-1  S/N:2452
  409130 like 201130 but mat=4094 rho=-6.070 u=409  $ ELEMENT D-9.4-1  S/N:2452
  409140 like 201140 but mat=4093 rho=-6.071 u=409  $ ELEMENT D-9.5-1  S/N:2452
  409150 like 201150 but mat=4092 rho=-6.072 u=409  $ ELEMENT D-9.6-1  S/N:2452
  409160 like 201160 but mat=4091 rho=-6.073 u=409  $ ELEMENT D-9.7-1  S/N:2452
c ==============================================================================
  410000 like 201000 but u=410
  410001 like 201001 but u=410
  410002 like 201002 but u=410
  410003 like 201003 but u=410
  410004 like 201004 but u=410
  410005 like 201005 but u=410
  410006 like 201006 but u=410
  410007 like 201007 but u=410
  410008 like 201008 but u=410
  410009 like 201009 but u=410
  410010 like 201010 but u=410
  410100 like 201100 but mat=4101 rho=-6.407 u=410  $ ELEMENT D-10.1-1  S/N:10893
  410110 like 201110 but mat=4102 rho=-6.407 u=410  $ ELEMENT D-10.2-1  S/N:10893
  410120 like 201120 but mat=4103 rho=-6.407 u=410  $ ELEMENT D-10.3-1  S/N:10893
  410130 like 201130 but mat=4104 rho=-6.407 u=410  $ ELEMENT D-10.4-1  S/N:10893
  410140 like 201140 but mat=4103 rho=-6.407 u=410  $ ELEMENT D-10.5-1  S/N:10893
  410150 like 201150 but mat=4102 rho=-6.407 u=410  $ ELEMENT D-10.6-1  S/N:10893
  410160 like 201160 but mat=4101 rho=-6.407 u=410  $ ELEMENT D-10.7-1  S/N:10893
c ==============================================================================
  411000 like 201000 but u=411
  411001 like 201001 but u=411
  411002 like 201002 but u=411
  411003 like 201003 but u=411
  411004 like 201004 but u=411
  411005 like 201005 but u=411
  411006 like 201006 but u=411
  411007 like 201007 but u=411
  411008 like 201008 but u=411
  411009 like 201009 but u=411
  411010 like 201010 but u=411
  411100 like 201100 but mat=4111 rho=-5.940 u=411  $ ELEMENT D-11.1-1  S/N:2435
  411110 like 201110 but mat=4112 rho=-5.939 u=411  $ ELEMENT D-11.2-1  S/N:2435
  411120 like 201120 but mat=4113 rho=-5.938 u=411  $ ELEMENT D-11.3-1  S/N:2435
  411130 like 201130 but mat=4114 rho=-5.937 u=411  $ ELEMENT D-11.4-1  S/N:2435
  411140 like 201140 but mat=4113 rho=-5.938 u=411  $ ELEMENT D-11.5-1  S/N:2435
  411150 like 201150 but mat=4112 rho=-5.939 u=411  $ ELEMENT D-11.6-1  S/N:2435
  411160 like 201160 but mat=4111 rho=-5.940 u=411  $ ELEMENT D-11.7-1  S/N:2435
c ==============================================================================
  412000 like 201000 but u=412
  412001 like 201001 but u=412
  412002 like 201002 but u=412
  412003 like 201003 but u=412
  412004 like 201004 but u=412
  412005 like 201005 but u=412
  412006 like 201006 but u=412
  412007 like 201007 but u=412
  412008 like 201008 but u=412
  412009 like 201009 but u=412
  412010 like 201010 but u=412
  412100 like 201100 but mat=4121 rho=-6.237 u=412  $ ELEMENT D-12.1-1  S/N:3876
  412110 like 201110 but mat=4122 rho=-6.236 u=412  $ ELEMENT D-12.2-1  S/N:3876
  412120 like 201120 but mat=4123 rho=-6.235 u=412  $ ELEMENT D-12.3-1  S/N:3876
  412130 like 201130 but mat=4124 rho=-6.234 u=412  $ ELEMENT D-12.4-1  S/N:3876
  412140 like 201140 but mat=4123 rho=-6.235 u=412  $ ELEMENT D-12.5-1  S/N:3876
  412150 like 201150 but mat=4122 rho=-6.236 u=412  $ ELEMENT D-12.6-1  S/N:3876
  412160 like 201160 but mat=4121 rho=-6.237 u=412  $ ELEMENT D-12.7-1  S/N:3876
c ==============================================================================
  413000 like 201000 but u=413
  413001 like 201001 but u=413
  413002 like 201002 but u=413
  413003 like 201003 but u=413
  413004 like 201004 but u=413
  413005 like 201005 but u=413
  413006 like 201006 but u=413
  413007 like 201007 but u=413
  413008 like 201008 but u=413
  413009 like 201009 but u=413
  413010 like 201010 but u=413
  413100 like 201100 but mat=4131 rho=-6.073 u=413  $ ELEMENT D-13.1-1  S/N:3696
  413110 like 201110 but mat=4132 rho=-6.072 u=413  $ ELEMENT D-13.2-1  S/N:3696
  413120 like 201120 but mat=4133 rho=-6.071 u=413  $ ELEMENT D-13.3-1  S/N:3696
  413130 like 201130 but mat=4134 rho=-6.071 u=413  $ ELEMENT D-13.4-1  S/N:3696
  413140 like 201140 but mat=4133 rho=-6.071 u=413  $ ELEMENT D-13.5-1  S/N:3696
  413150 like 201150 but mat=4132 rho=-6.072 u=413  $ ELEMENT D-13.6-1  S/N:3696
  413160 like 201160 but mat=4131 rho=-6.073 u=413  $ ELEMENT D-13.7-1  S/N:3696
c ==============================================================================
  414000 like 201000 but u=414
  414001 like 201001 but u=414
  414002 like 201002 but u=414
  414003 like 201003 but u=414
  414004 like 201004 but u=414
  414005 like 201005 but u=414
  414006 like 201006 but u=414
  414007 like 201007 but u=414
  414008 like 201008 but u=414
  414009 like 201009 but u=414
  414010 like 201010 but u=414
  414100 like 201100 but mat=4141 rho=-6.402 u=414  $ ELEMENT D-14.1-1  S/N:3494
  414110 like 201110 but mat=4142 rho=-6.401 u=414  $ ELEMENT D-14.2-1  S/N:3494
  414120 like 201120 but mat=4143 rho=-6.399 u=414  $ ELEMENT D-14.3-1  S/N:3494
  414130 like 201130 but mat=4144 rho=-6.399 u=414  $ ELEMENT D-14.4-1  S/N:3494
  414140 like 201140 but mat=4143 rho=-6.399 u=414  $ ELEMENT D-14.5-1  S/N:3494
  414150 like 201150 but mat=4142 rho=-6.401 u=414  $ ELEMENT D-14.6-1  S/N:3494
  414160 like 201160 but mat=4141 rho=-6.402 u=414  $ ELEMENT D-14.7-1  S/N:3494
c ==============================================================================
  415000 like 201000 but u=415
  415001 like 201001 but u=415
  415002 like 201002 but u=415
  415003 like 201003 but u=415
  415004 like 201004 but u=415
  415005 like 201005 but u=415
  415006 like 201006 but u=415
  415007 like 201007 but u=415
  415008 like 201008 but u=415
  415009 like 201009 but u=415
  415010 like 201010 but u=415
  415100 like 201100 but mat=4151 rho=-6.237 u=415  $ ELEMENT D-15.1-1  S/N:3501
  415110 like 201110 but mat=4152 rho=-6.236 u=415  $ ELEMENT D-15.2-1  S/N:3501
  415120 like 201120 but mat=4153 rho=-6.235 u=415  $ ELEMENT D-15.3-1  S/N:3501
  415130 like 201130 but mat=4154 rho=-6.235 u=415  $ ELEMENT D-15.4-1  S/N:3501
  415140 like 201140 but mat=4153 rho=-6.235 u=415  $ ELEMENT D-15.5-1  S/N:3501
  415150 like 201150 but mat=4152 rho=-6.236 u=415  $ ELEMENT D-15.6-1  S/N:3501
  415160 like 201160 but mat=4151 rho=-6.237 u=415  $ ELEMENT D-15.7-1  S/N:3501
c ==============================================================================
  417000 like 201000 but u=417
  417001 like 201001 but u=417
  417002 like 201002 but u=417
  417003 like 201003 but u=417
  417004 like 201004 but u=417
  417005 like 201005 but u=417
  417006 like 201006 but u=417
  417007 like 201007 but u=417
  417008 like 201008 but u=417
  417009 like 201009 but u=417
  417010 like 201010 but u=417
  417100 like 201100 but mat=4171 rho=-6.237 u=417  $ ELEMENT D-17.1-1  S/N:3498
  417110 like 201110 but mat=4172 rho=-6.236 u=417  $ ELEMENT D-17.2-1  S/N:3498
  417120 like 201120 but mat=4173 rho=-6.235 u=417  $ ELEMENT D-17.3-1  S/N:3498
  417130 like 201130 but mat=4174 rho=-6.235 u=417  $ ELEMENT D-17.4-1  S/N:3498
  417140 like 201140 but mat=4173 rho=-6.235 u=417  $ ELEMENT D-17.5-1  S/N:3498
  417150 like 201150 but mat=4172 rho=-6.236 u=417  $ ELEMENT D-17.6-1  S/N:3498
  417160 like 201160 but mat=4171 rho=-6.237 u=417  $ ELEMENT D-17.7-1  S/N:3498
c ==============================================================================
  418000 like 201000 but u=418
  418001 like 201001 but u=418
  418002 like 201002 but u=418
  418003 like 201003 but u=418
  418004 like 201004 but u=418
  418005 like 201005 but u=418
  418006 like 201006 but u=418
  418007 like 201007 but u=418
  418008 like 201008 but u=418
  418009 like 201009 but u=418
  418010 like 201010 but u=418
  418100 like 201100 but mat=4181 rho=-6.073 u=418  $ ELEMENT D-18.1-1  S/N:3336
  418110 like 201110 but mat=4182 rho=-6.072 u=418  $ ELEMENT D-18.2-1  S/N:3336
  418120 like 201120 but mat=4183 rho=-6.071 u=418  $ ELEMENT D-18.3-1  S/N:3336
  418130 like 201130 but mat=4184 rho=-6.071 u=418  $ ELEMENT D-18.4-1  S/N:3336
  418140 like 201140 but mat=4183 rho=-6.071 u=418  $ ELEMENT D-18.5-1  S/N:3336
  418150 like 201150 but mat=4182 rho=-6.072 u=418  $ ELEMENT D-18.6-1  S/N:3336
  418160 like 201160 but mat=4181 rho=-6.073 u=418  $ ELEMENT D-18.7-1  S/N:3336
c ==============================================================================
  502000 like 201000 but u=502
  502001 like 201001 but u=502
  502002 like 201002 but u=502
  502003 like 201003 but u=502
  502004 like 201004 but u=502
  502005 like 201005 but u=502
  502006 like 201006 but u=502
  502007 like 201007 but u=502
  502008 like 201008 but u=502
  502009 like 201009 but u=502
  502010 like 201010 but u=502
  502100 like 201100 but mat=5021 rho=-6.327 u=502  $ ELEMENT E-2.1-1  S/N:2986
  502110 like 201110 but mat=5022 rho=-6.327 u=502  $ ELEMENT E-2.2-1  S/N:2986
  502120 like 201120 but mat=5023 rho=-6.326 u=502  $ ELEMENT E-2.3-1  S/N:2986
  502130 like 201130 but mat=5024 rho=-6.326 u=502  $ ELEMENT E-2.4-1  S/N:2986
  502140 like 201140 but mat=5023 rho=-6.326 u=502  $ ELEMENT E-2.5-1  S/N:2986
  502150 like 201150 but mat=5022 rho=-6.327 u=502  $ ELEMENT E-2.6-1  S/N:2986
  502160 like 201160 but mat=5021 rho=-6.327 u=502  $ ELEMENT E-2.7-1  S/N:2986
c ==============================================================================
  503000 like 201000 but u=503
  503001 like 201001 but u=503
  503002 like 201002 but u=503
  503003 like 201003 but u=503
  503004 like 201004 but u=503
  503005 like 201005 but u=503
  503006 like 201006 but u=503
  503007 like 201007 but u=503
  503008 like 201008 but u=503
  503009 like 201009 but u=503
  503010 like 201010 but u=503
  503100 like 201100 but mat=5031 rho=-5.939 u=503  $ ELEMENT E-3.1-1  S/N:2458
  503110 like 201110 but mat=5032 rho=-5.938 u=503  $ ELEMENT E-3.2-1  S/N:2458
  503120 like 201120 but mat=5033 rho=-5.937 u=503  $ ELEMENT E-3.3-1  S/N:2458
  503130 like 201130 but mat=5034 rho=-5.937 u=503  $ ELEMENT E-3.4-1  S/N:2458
  503140 like 201140 but mat=5033 rho=-5.937 u=503  $ ELEMENT E-3.5-1  S/N:2458
  503150 like 201150 but mat=5032 rho=-5.938 u=503  $ ELEMENT E-3.6-1  S/N:2458
  503160 like 201160 but mat=5031 rho=-5.939 u=503  $ ELEMENT E-3.7-1  S/N:2458
c ==============================================================================
  504000 like 201000 but u=504
  504001 like 201001 but u=504
  504002 like 201002 but u=504
  504003 like 201003 but u=504
  504004 like 201004 but u=504
  504005 like 201005 but u=504
  504006 like 201006 but u=504
  504007 like 201007 but u=504
  504008 like 201008 but u=504
  504009 like 201009 but u=504
  504010 like 201010 but u=504
  504100 like 201100 but mat=5041 rho=-6.074 u=504  $ ELEMENT E-4.1-1  S/N:3006
  504110 like 201110 but mat=5042 rho=-6.072 u=504  $ ELEMENT E-4.2-1  S/N:3006
  504120 like 201120 but mat=5043 rho=-6.071 u=504  $ ELEMENT E-4.3-1  S/N:3006
  504130 like 201130 but mat=5044 rho=-6.071 u=504  $ ELEMENT E-4.4-1  S/N:3006
  504140 like 201140 but mat=5043 rho=-6.071 u=504  $ ELEMENT E-4.5-1  S/N:3006
  504150 like 201150 but mat=5042 rho=-6.072 u=504  $ ELEMENT E-4.6-1  S/N:3006
  504160 like 201160 but mat=5041 rho=-6.074 u=504  $ ELEMENT E-4.7-1  S/N:3006
c ==============================================================================
  505000 like 201000 but u=505
  505001 like 201001 but u=505
  505002 like 201002 but u=505
  505003 like 201003 but u=505
  505004 like 201004 but u=505
  505005 like 201005 but u=505
  505006 like 201006 but u=505
  505007 like 201007 but u=505
  505008 like 201008 but u=505
  505009 like 201009 but u=505
  505010 like 201010 but u=505
  505100 like 201100 but mat=5051 rho=-6.235 u=505  $ ELEMENT E-5.1-1  S/N:5014
  505110 like 201110 but mat=5052 rho=-6.233 u=505  $ ELEMENT E-5.2-1  S/N:5014
  505120 like 201120 but mat=5053 rho=-6.231 u=505  $ ELEMENT E-5.3-1  S/N:5014
  505130 like 201130 but mat=5054 rho=-6.230 u=505  $ ELEMENT E-5.4-1  S/N:5014
  505140 like 201140 but mat=5053 rho=-6.231 u=505  $ ELEMENT E-5.5-1  S/N:5014
  505150 like 201150 but mat=5052 rho=-6.233 u=505  $ ELEMENT E-5.6-1  S/N:5014
  505160 like 201160 but mat=5051 rho=-6.235 u=505  $ ELEMENT E-5.7-1  S/N:5014
c ==============================================================================
  506000 like 201000 but u=506
  506001 like 201001 but u=506
  506002 like 201002 but u=506
  506003 like 201003 but u=506
  506004 like 201004 but u=506
  506005 like 201005 but u=506
  506006 like 201006 but u=506
  506007 like 201007 but u=506
  506008 like 201008 but u=506
  506009 like 201009 but u=506
  506010 like 201010 but u=506
  506100 like 201100 but mat=5061 rho=-6.235 u=506  $ ELEMENT E-6.1-1  S/N:4744
  506110 like 201110 but mat=5062 rho=-6.233 u=506  $ ELEMENT E-6.2-1  S/N:4744
  506120 like 201120 but mat=5063 rho=-6.231 u=506  $ ELEMENT E-6.3-1  S/N:4744
  506130 like 201130 but mat=5064 rho=-6.231 u=506  $ ELEMENT E-6.4-1  S/N:4744
  506140 like 201140 but mat=5063 rho=-6.231 u=506  $ ELEMENT E-6.5-1  S/N:4744
  506150 like 201150 but mat=5062 rho=-6.233 u=506  $ ELEMENT E-6.6-1  S/N:4744
  506160 like 201160 but mat=5061 rho=-6.235 u=506  $ ELEMENT E-6.7-1  S/N:4744
c ==============================================================================
  507000 like 201000 but u=507
  507001 like 201001 but u=507
  507002 like 201002 but u=507
  507003 like 201003 but u=507
  507004 like 201004 but u=507
  507005 like 201005 but u=507
  507006 like 201006 but u=507
  507007 like 201007 but u=507
  507008 like 201008 but u=507
  507009 like 201009 but u=507
  507010 like 201010 but u=507
  507100 like 201100 but mat=5071 rho=-5.938 u=507  $ ELEMENT E-7.1-1  S/N:3147
  507110 like 201110 but mat=5072 rho=-5.937 u=507  $ ELEMENT E-7.2-1  S/N:3147
  507120 like 201120 but mat=5073 rho=-5.935 u=507  $ ELEMENT E-7.3-1  S/N:3147
  507130 like 201130 but mat=5074 rho=-5.935 u=507  $ ELEMENT E-7.4-1  S/N:3147
  507140 like 201140 but mat=5073 rho=-5.935 u=507  $ ELEMENT E-7.5-1  S/N:3147
  507150 like 201150 but mat=5072 rho=-5.937 u=507  $ ELEMENT E-7.6-1  S/N:3147
  507160 like 201160 but mat=5071 rho=-5.938 u=507  $ ELEMENT E-7.7-1  S/N:3147
c ==============================================================================
  508000 like 201000 but u=508
  508001 like 201001 but u=508
  508002 like 201002 but u=508
  508003 like 201003 but u=508
  508004 like 201004 but u=508
  508005 like 201005 but u=508
  508006 like 201006 but u=508
  508007 like 201007 but u=508
  508008 like 201008 but u=508
  508009 like 201009 but u=508
  508010 like 201010 but u=508
  508100 like 201100 but mat=5081 rho=-6.235 u=508  $ ELEMENT E-8.1-1  S/N:4991
  508110 like 201110 but mat=5082 rho=-6.233 u=508  $ ELEMENT E-8.2-1  S/N:4991
  508120 like 201120 but mat=5083 rho=-6.231 u=508  $ ELEMENT E-8.3-1  S/N:4991
  508130 like 201130 but mat=5084 rho=-6.230 u=508  $ ELEMENT E-8.4-1  S/N:4991
  508140 like 201140 but mat=5083 rho=-6.231 u=508  $ ELEMENT E-8.5-1  S/N:4991
  508150 like 201150 but mat=5082 rho=-6.233 u=508  $ ELEMENT E-8.6-1  S/N:4991
  508160 like 201160 but mat=5081 rho=-6.235 u=508  $ ELEMENT E-8.7-1  S/N:4991
c ==============================================================================
  509000 like 201000 but u=509
  509001 like 201001 but u=509
  509002 like 201002 but u=509
  509003 like 201003 but u=509
  509004 like 201004 but u=509
  509005 like 201005 but u=509
  509006 like 201006 but u=509
  509007 like 201007 but u=509
  509008 like 201008 but u=509
  509009 like 201009 but u=509
  509010 like 201010 but u=509
  509100 like 201100 but mat=5091 rho=-6.231 u=509  $ ELEMENT E-9.1-1  S/N:4742
  509110 like 201110 but mat=5092 rho=-6.228 u=509  $ ELEMENT E-9.2-1  S/N:4742
  509120 like 201120 but mat=5093 rho=-6.226 u=509  $ ELEMENT E-9.3-1  S/N:4742
  509130 like 201130 but mat=5094 rho=-6.225 u=509  $ ELEMENT E-9.4-1  S/N:4742
  509140 like 201140 but mat=5093 rho=-6.226 u=509  $ ELEMENT E-9.5-1  S/N:4742
  509150 like 201150 but mat=5092 rho=-6.228 u=509  $ ELEMENT E-9.6-1  S/N:4742
  509160 like 201160 but mat=5091 rho=-6.231 u=509  $ ELEMENT E-9.7-1  S/N:4742
c ==============================================================================
  510000 like 201000 but u=510
  510001 like 201001 but u=510
  510002 like 201002 but u=510
  510003 like 201003 but u=510
  510004 like 201004 but u=510
  510005 like 201005 but u=510
  510006 like 201006 but u=510
  510007 like 201007 but u=510
  510008 like 201008 but u=510
  510009 like 201009 but u=510
  510010 like 201010 but u=510
  510100 like 201100 but mat=5101 rho=-6.397 u=510  $ ELEMENT E-10.1-1  S/N:4351
  510110 like 201110 but mat=5102 rho=-6.395 u=510  $ ELEMENT E-10.2-1  S/N:4351
  510120 like 201120 but mat=5103 rho=-6.392 u=510  $ ELEMENT E-10.3-1  S/N:4351
  510130 like 201130 but mat=5104 rho=-6.392 u=510  $ ELEMENT E-10.4-1  S/N:4351
  510140 like 201140 but mat=5103 rho=-6.392 u=510  $ ELEMENT E-10.5-1  S/N:4351
  510150 like 201150 but mat=5102 rho=-6.395 u=510  $ ELEMENT E-10.6-1  S/N:4351
  510160 like 201160 but mat=5101 rho=-6.397 u=510  $ ELEMENT E-10.7-1  S/N:4351
c ==============================================================================
  511000 like 201000 but u=511
  511001 like 201001 but u=511
  511002 like 201002 but u=511
  511003 like 201003 but u=511
  511004 like 201004 but u=511
  511005 like 201005 but u=511
  511006 like 201006 but u=511
  511007 like 201007 but u=511
  511008 like 201008 but u=511
  511009 like 201009 but u=511
  511010 like 201010 but u=511
  511100 like 201100 but mat=5111 rho=-6.072 u=511  $ ELEMENT E-11.1-1  S/N:3107
  511110 like 201110 but mat=5112 rho=-6.070 u=511  $ ELEMENT E-11.2-1  S/N:3107
  511120 like 201120 but mat=5113 rho=-6.069 u=511  $ ELEMENT E-11.3-1  S/N:3107
  511130 like 201130 but mat=5114 rho=-6.069 u=511  $ ELEMENT E-11.4-1  S/N:3107
  511140 like 201140 but mat=5113 rho=-6.069 u=511  $ ELEMENT E-11.5-1  S/N:3107
  511150 like 201150 but mat=5112 rho=-6.070 u=511  $ ELEMENT E-11.6-1  S/N:3107
  511160 like 201160 but mat=5111 rho=-6.072 u=511  $ ELEMENT E-11.7-1  S/N:3107
c ==============================================================================
  512000 like 201000 but u=512
  512001 like 201001 but u=512
  512002 like 201002 but u=512
  512003 like 201003 but u=512
  512004 like 201004 but u=512
  512005 like 201005 but u=512
  512006 like 201006 but u=512
  512007 like 201007 but u=512
  512008 like 201008 but u=512
  512009 like 201009 but u=512
  512010 like 201010 but u=512
  512100 like 201100 but mat=5121 rho=-6.238 u=512  $ ELEMENT E-12.1-1  S/N:3690
  512110 like 201110 but mat=5122 rho=-6.237 u=512  $ ELEMENT E-12.2-1  S/N:3690
  512120 like 201120 but mat=5123 rho=-6.236 u=512  $ ELEMENT E-12.3-1  S/N:3690
  512130 like 201130 but mat=5124 rho=-6.235 u=512  $ ELEMENT E-12.4-1  S/N:3690
  512140 like 201140 but mat=5123 rho=-6.236 u=512  $ ELEMENT E-12.5-1  S/N:3690
  512150 like 201150 but mat=5122 rho=-6.237 u=512  $ ELEMENT E-12.6-1  S/N:3690
  512160 like 201160 but mat=5121 rho=-6.238 u=512  $ ELEMENT E-12.7-1  S/N:3690
c ==============================================================================
  513000 like 201000 but u=513
  513001 like 201001 but u=513
  513002 like 201002 but u=513
  513003 like 201003 but u=513
  513004 like 201004 but u=513
  513005 like 201005 but u=513
  513006 like 201006 but u=513
  513007 like 201007 but u=513
  513008 like 201008 but u=513
  513009 like 201009 but u=513
  513010 like 201010 but u=513
  513100 like 201100 but mat=5131 rho=-6.370 u=513  $ ELEMENT E-13.1-1  S/N:2987
  513110 like 201110 but mat=5132 rho=-6.370 u=513  $ ELEMENT E-13.2-1  S/N:2987
  513120 like 201120 but mat=5133 rho=-6.369 u=513  $ ELEMENT E-13.3-1  S/N:2987
  513130 like 201130 but mat=5134 rho=-6.369 u=513  $ ELEMENT E-13.4-1  S/N:2987
  513140 like 201140 but mat=5133 rho=-6.369 u=513  $ ELEMENT E-13.5-1  S/N:2987
  513150 like 201150 but mat=5132 rho=-6.370 u=513  $ ELEMENT E-13.6-1  S/N:2987
  513160 like 201160 but mat=5131 rho=-6.370 u=513  $ ELEMENT E-13.7-1  S/N:2987
c ==============================================================================
  514000 like 201000 but u=514
  514001 like 201001 but u=514
  514002 like 201002 but u=514
  514003 like 201003 but u=514
  514004 like 201004 but u=514
  514005 like 201005 but u=514
  514006 like 201006 but u=514
  514007 like 201007 but u=514
  514008 like 201008 but u=514
  514009 like 201009 but u=514
  514010 like 201010 but u=514
  514100 like 201100 but mat=5141 rho=-6.073 u=514  $ ELEMENT E-14.1-1  S/N:3118
  514110 like 201110 but mat=5142 rho=-6.072 u=514  $ ELEMENT E-14.2-1  S/N:3118
  514120 like 201120 but mat=5143 rho=-6.070 u=514  $ ELEMENT E-14.3-1  S/N:3118
  514130 like 201130 but mat=5144 rho=-6.070 u=514  $ ELEMENT E-14.4-1  S/N:3118
  514140 like 201140 but mat=5143 rho=-6.070 u=514  $ ELEMENT E-14.5-1  S/N:3118
  514150 like 201150 but mat=5142 rho=-6.072 u=514  $ ELEMENT E-14.6-1  S/N:3118
  514160 like 201160 but mat=5141 rho=-6.073 u=514  $ ELEMENT E-14.7-1  S/N:3118
c ==============================================================================
  515000 like 201000 but u=515
  515001 like 201001 but u=515
  515002 like 201002 but u=515
  515003 like 201003 but u=515
  515004 like 201004 but u=515
  515005 like 201005 but u=515
  515006 like 201006 but u=515
  515007 like 201007 but u=515
  515008 like 201008 but u=515
  515009 like 201009 but u=515
  515010 like 201010 but u=515
  515100 like 201100 but mat=5151 rho=-6.385 u=515  $ ELEMENT E-15.1-1  S/N:2934
  515110 like 201110 but mat=5152 rho=-6.384 u=515  $ ELEMENT E-15.2-1  S/N:2934
  515120 like 201120 but mat=5153 rho=-6.384 u=515  $ ELEMENT E-15.3-1  S/N:2934
  515130 like 201130 but mat=5154 rho=-6.384 u=515  $ ELEMENT E-15.4-1  S/N:2934
  515140 like 201140 but mat=5153 rho=-6.384 u=515  $ ELEMENT E-15.5-1  S/N:2934
  515150 like 201150 but mat=5152 rho=-6.384 u=515  $ ELEMENT E-15.6-1  S/N:2934
  515160 like 201160 but mat=5151 rho=-6.385 u=515  $ ELEMENT E-15.7-1  S/N:2934
c ==============================================================================
  516000 like 201000 but u=516
  516001 like 201001 but u=516
  516002 like 201002 but u=516
  516003 like 201003 but u=516
  516004 like 201004 but u=516
  516005 like 201005 but u=516
  516006 like 201006 but u=516
  516007 like 201007 but u=516
  516008 like 201008 but u=516
  516009 like 201009 but u=516
  516010 like 201010 but u=516
  516100 like 201100 but mat=5161 rho=-6.234 u=516  $ ELEMENT E-16.1-1  S/N:4343
  516110 like 201110 but mat=5162 rho=-6.232 u=516  $ ELEMENT E-16.2-1  S/N:4343
  516120 like 201120 but mat=5163 rho=-6.230 u=516  $ ELEMENT E-16.3-1  S/N:4343
  516130 like 201130 but mat=5164 rho=-6.230 u=516  $ ELEMENT E-16.4-1  S/N:4343
  516140 like 201140 but mat=5163 rho=-6.230 u=516  $ ELEMENT E-16.5-1  S/N:4343
  516150 like 201150 but mat=5162 rho=-6.232 u=516  $ ELEMENT E-16.6-1  S/N:4343
  516160 like 201160 but mat=5161 rho=-6.234 u=516  $ ELEMENT E-16.7-1  S/N:4343
c ==============================================================================
  517000 like 201000 but u=517
  517001 like 201001 but u=517
  517002 like 201002 but u=517
  517003 like 201003 but u=517
  517004 like 201004 but u=517
  517005 like 201005 but u=517
  517006 like 201006 but u=517
  517007 like 201007 but u=517
  517008 like 201008 but u=517
  517009 like 201009 but u=517
  517010 like 201010 but u=517
  517100 like 201100 but mat=5171 rho=-6.239 u=517  $ ELEMENT E-17.1-1  S/N:3517
  517110 like 201110 but mat=5172 rho=-6.239 u=517  $ ELEMENT E-17.2-1  S/N:3517
  517120 like 201120 but mat=5173 rho=-6.238 u=517  $ ELEMENT E-17.3-1  S/N:3517
  517130 like 201130 but mat=5174 rho=-6.238 u=517  $ ELEMENT E-17.4-1  S/N:3517
  517140 like 201140 but mat=5173 rho=-6.238 u=517  $ ELEMENT E-17.5-1  S/N:3517
  517150 like 201150 but mat=5172 rho=-6.239 u=517  $ ELEMENT E-17.6-1  S/N:3517
  517160 like 201160 but mat=5171 rho=-6.239 u=517  $ ELEMENT E-17.7-1  S/N:3517
c ==============================================================================
  518000 like 201000 but u=518
  518001 like 201001 but u=518
  518002 like 201002 but u=518
  518003 like 201003 but u=518
  518004 like 201004 but u=518
  518005 like 201005 but u=518
  518006 like 201006 but u=518
  518007 like 201007 but u=518
  518008 like 201008 but u=518
  518009 like 201009 but u=518
  518010 like 201010 but u=518
  518100 like 201100 but mat=5181 rho=-6.075 u=518  $ ELEMENT E-18.1-1  S/N:3502
  518110 like 201110 but mat=5182 rho=-6.074 u=518  $ ELEMENT E-18.2-1  S/N:3502
  518120 like 201120 but mat=5183 rho=-6.074 u=518  $ ELEMENT E-18.3-1  S/N:3502
  518130 like 201130 but mat=5184 rho=-6.073 u=518  $ ELEMENT E-18.4-1  S/N:3502
  518140 like 201140 but mat=5183 rho=-6.074 u=518  $ ELEMENT E-18.5-1  S/N:3502
  518150 like 201150 but mat=5182 rho=-6.074 u=518  $ ELEMENT E-18.6-1  S/N:3502
  518160 like 201160 but mat=5181 rho=-6.075 u=518  $ ELEMENT E-18.7-1  S/N:3502
c ==============================================================================
  519000 like 201000 but u=519
  519001 like 201001 but u=519
  519002 like 201002 but u=519
  519003 like 201003 but u=519
  519004 like 201004 but u=519
  519005 like 201005 but u=519
  519006 like 201006 but u=519
  519007 like 201007 but u=519
  519008 like 201008 but u=519
  519009 like 201009 but u=519
  519010 like 201010 but u=519
  519100 like 201100 but mat=5191 rho=-6.238 u=519  $ ELEMENT E-19.1-1  S/N:3503
  519110 like 201110 but mat=5192 rho=-6.237 u=519  $ ELEMENT E-19.2-1  S/N:3503
  519120 like 201120 but mat=5193 rho=-6.236 u=519  $ ELEMENT E-19.3-1  S/N:3503
  519130 like 201130 but mat=5194 rho=-6.235 u=519  $ ELEMENT E-19.4-1  S/N:3503
  519140 like 201140 but mat=5193 rho=-6.236 u=519  $ ELEMENT E-19.5-1  S/N:3503
  519150 like 201150 but mat=5192 rho=-6.237 u=519  $ ELEMENT E-19.6-1  S/N:3503
  519160 like 201160 but mat=5191 rho=-6.238 u=519  $ ELEMENT E-19.7-1  S/N:3503
c ==============================================================================
  520000 like 201000 but u=520
  520001 like 201001 but u=520
  520002 like 201002 but u=520
  520003 like 201003 but u=520
  520004 like 201004 but u=520
  520005 like 201005 but u=520
  520006 like 201006 but u=520
  520007 like 201007 but u=520
  520008 like 201008 but u=520
  520009 like 201009 but u=520
  520010 like 201010 but u=520
  520100 like 201100 but mat=5201 rho=-6.237 u=520  $ ELEMENT E-20.1-1  S/N:3009
  520110 like 201110 but mat=5202 rho=-6.236 u=520  $ ELEMENT E-20.2-1  S/N:3009
  520120 like 201120 but mat=5203 rho=-6.235 u=520  $ ELEMENT E-20.3-1  S/N:3009
  520130 like 201130 but mat=5204 rho=-6.234 u=520  $ ELEMENT E-20.4-1  S/N:3009
  520140 like 201140 but mat=5203 rho=-6.235 u=520  $ ELEMENT E-20.5-1  S/N:3009
  520150 like 201150 but mat=5202 rho=-6.236 u=520  $ ELEMENT E-20.6-1  S/N:3009
  520160 like 201160 but mat=5201 rho=-6.237 u=520  $ ELEMENT E-20.7-1  S/N:3009
c ==============================================================================
  521000 like 201000 but u=521
  521001 like 201001 but u=521
  521002 like 201002 but u=521
  521003 like 201003 but u=521
  521004 like 201004 but u=521
  521005 like 201005 but u=521
  521006 like 201006 but u=521
  521007 like 201007 but u=521
  521008 like 201008 but u=521
  521009 like 201009 but u=521
  521010 like 201010 but u=521
  521100 like 201100 but mat=5211 rho=-6.237 u=521  $ ELEMENT E-21.1-1  S/N:3011
  521110 like 201110 but mat=5212 rho=-6.235 u=521  $ ELEMENT E-21.2-1  S/N:3011
  521120 like 201120 but mat=5213 rho=-6.234 u=521  $ ELEMENT E-21.3-1  S/N:3011
  521130 like 201130 but mat=5214 rho=-6.234 u=521  $ ELEMENT E-21.4-1  S/N:3011
  521140 like 201140 but mat=5213 rho=-6.234 u=521  $ ELEMENT E-21.5-1  S/N:3011
  521150 like 201150 but mat=5212 rho=-6.235 u=521  $ ELEMENT E-21.6-1  S/N:3011
  521160 like 201160 but mat=5211 rho=-6.237 u=521  $ ELEMENT E-21.7-1  S/N:3011
c ==============================================================================
  522000 like 201000 but u=522
  522001 like 201001 but u=522
  522002 like 201002 but u=522
  522003 like 201003 but u=522
  522004 like 201004 but u=522
  522005 like 201005 but u=522
  522006 like 201006 but u=522
  522007 like 201007 but u=522
  522008 like 201008 but u=522
  522009 like 201009 but u=522
  522010 like 201010 but u=522
  522100 like 201100 but mat=5221 rho=-6.237 u=522  $ ELEMENT E-22.1-1  S/N:3014
  522110 like 201110 but mat=5222 rho=-6.235 u=522  $ ELEMENT E-22.2-1  S/N:3014
  522120 like 201120 but mat=5223 rho=-6.234 u=522  $ ELEMENT E-22.3-1  S/N:3014
  522130 like 201130 but mat=5224 rho=-6.234 u=522  $ ELEMENT E-22.4-1  S/N:3014
  522140 like 201140 but mat=5223 rho=-6.234 u=522  $ ELEMENT E-22.5-1  S/N:3014
  522150 like 201150 but mat=5222 rho=-6.235 u=522  $ ELEMENT E-22.6-1  S/N:3014
  522160 like 201160 but mat=5221 rho=-6.237 u=522  $ ELEMENT E-22.7-1  S/N:3014
c ==============================================================================
  523000 like 201000 but u=523
  523001 like 201001 but u=523
  523002 like 201002 but u=523
  523003 like 201003 but u=523
  523004 like 201004 but u=523
  523005 like 201005 but u=523
  523006 like 201006 but u=523
  523007 like 201007 but u=523
  523008 like 201008 but u=523
  523009 like 201009 but u=523
  523010 like 201010 but u=523
  523100 like 201100 but mat=5231 rho=-6.073 u=523  $ ELEMENT E-23.1-1  S/N:3008
  523110 like 201110 but mat=5232 rho=-6.072 u=523  $ ELEMENT E-23.2-1  S/N:3008
  523120 like 201120 but mat=5233 rho=-6.071 u=523  $ ELEMENT E-23.3-1  S/N:3008
  523130 like 201130 but mat=5234 rho=-6.071 u=523  $ ELEMENT E-23.4-1  S/N:3008
  523140 like 201140 but mat=5233 rho=-6.071 u=523  $ ELEMENT E-23.5-1  S/N:3008
  523150 like 201150 but mat=5232 rho=-6.072 u=523  $ ELEMENT E-23.6-1  S/N:3008
  523160 like 201160 but mat=5231 rho=-6.073 u=523  $ ELEMENT E-23.7-1  S/N:3008
c ==============================================================================
  524000 like 201000 but u=524
  524001 like 201001 but u=524
  524002 like 201002 but u=524
  524003 like 201003 but u=524
  524004 like 201004 but u=524
  524005 like 201005 but u=524
  524006 like 201006 but u=524
  524007 like 201007 but u=524
  524008 like 201008 but u=524
  524009 like 201009 but u=524
  524010 like 201010 but u=524
  524100 like 201100 but mat=5241 rho=-6.073 u=524  $ ELEMENT E-24.1-1  S/N:3111
  524110 like 201110 but mat=5242 rho=-6.071 u=524  $ ELEMENT E-24.2-1  S/N:3111
  524120 like 201120 but mat=5243 rho=-6.070 u=524  $ ELEMENT E-24.3-1  S/N:3111
  524130 like 201130 but mat=5244 rho=-6.070 u=524  $ ELEMENT E-24.4-1  S/N:3111
  524140 like 201140 but mat=5243 rho=-6.070 u=524  $ ELEMENT E-24.5-1  S/N:3111
  524150 like 201150 but mat=5242 rho=-6.071 u=524  $ ELEMENT E-24.6-1  S/N:3111
  524160 like 201160 but mat=5241 rho=-6.073 u=524  $ ELEMENT E-24.7-1  S/N:3111
c ==============================================================================
  601000 like 201000 but u=601
  601001 like 201001 but u=601
  601002 like 201002 but u=601
  601003 like 201003 but u=601
  601004 like 201004 but u=601
  601005 like 201005 but u=601
  601006 like 201006 but u=601
  601007 like 201007 but u=601
  601008 like 201008 but u=601
  601009 like 201009 but u=601
  601010 like 201010 but u=601
  601100 like 201100 but mat=6011 rho=-6.234 u=601  $ ELEMENT F-1.1-1  S/N:5017
  601110 like 201110 but mat=6012 rho=-6.231 u=601  $ ELEMENT F-1.2-1  S/N:5017
  601120 like 201120 but mat=6013 rho=-6.229 u=601  $ ELEMENT F-1.3-1  S/N:5017
  601130 like 201130 but mat=6014 rho=-6.229 u=601  $ ELEMENT F-1.4-1  S/N:5017
  601140 like 201140 but mat=6013 rho=-6.229 u=601  $ ELEMENT F-1.5-1  S/N:5017
  601150 like 201150 but mat=6012 rho=-6.231 u=601  $ ELEMENT F-1.6-1  S/N:5017
  601160 like 201160 but mat=6011 rho=-6.234 u=601  $ ELEMENT F-1.7-1  S/N:5017
c ==============================================================================
  602000 like 201000 but u=602
  602001 like 201001 but u=602
  602002 like 201002 but u=602
  602003 like 201003 but u=602
  602004 like 201004 but u=602
  602005 like 201005 but u=602
  602006 like 201006 but u=602
  602007 like 201007 but u=602
  602008 like 201008 but u=602
  602009 like 201009 but u=602
  602010 like 201010 but u=602
  602100 like 201100 but mat=6021 rho=-6.234 u=602  $ ELEMENT F-2.1-1  S/N:5018
  602110 like 201110 but mat=6022 rho=-6.232 u=602  $ ELEMENT F-2.2-1  S/N:5018
  602120 like 201120 but mat=6023 rho=-6.230 u=602  $ ELEMENT F-2.3-1  S/N:5018
  602130 like 201130 but mat=6024 rho=-6.229 u=602  $ ELEMENT F-2.4-1  S/N:5018
  602140 like 201140 but mat=6023 rho=-6.230 u=602  $ ELEMENT F-2.5-1  S/N:5018
  602150 like 201150 but mat=6022 rho=-6.232 u=602  $ ELEMENT F-2.6-1  S/N:5018
  602160 like 201160 but mat=6021 rho=-6.234 u=602  $ ELEMENT F-2.7-1  S/N:5018
c ==============================================================================
  603000 like 201000 but u=603
  603001 like 201001 but u=603
  603002 like 201002 but u=603
  603003 like 201003 but u=603
  603004 like 201004 but u=603
  603005 like 201005 but u=603
  603006 like 201006 but u=603
  603007 like 201007 but u=603
  603008 like 201008 but u=603
  603009 like 201009 but u=603
  603010 like 201010 but u=603
  603100 like 201100 but mat=6031 rho=-6.398 u=603  $ ELEMENT F-3.1-1  S/N:5027
  603110 like 201110 but mat=6032 rho=-6.396 u=603  $ ELEMENT F-3.2-1  S/N:5027
  603120 like 201120 but mat=6033 rho=-6.394 u=603  $ ELEMENT F-3.3-1  S/N:5027
  603130 like 201130 but mat=6034 rho=-6.393 u=603  $ ELEMENT F-3.4-1  S/N:5027
  603140 like 201140 but mat=6033 rho=-6.394 u=603  $ ELEMENT F-3.5-1  S/N:5027
  603150 like 201150 but mat=6032 rho=-6.396 u=603  $ ELEMENT F-3.6-1  S/N:5027
  603160 like 201160 but mat=6031 rho=-6.398 u=603  $ ELEMENT F-3.7-1  S/N:5027
c ==============================================================================
  604000 like 201000 but u=604
  604001 like 201001 but u=604
  604002 like 201002 but u=604
  604003 like 201003 but u=604
  604004 like 201004 but u=604
  604005 like 201005 but u=604
  604006 like 201006 but u=604
  604007 like 201007 but u=604
  604008 like 201008 but u=604
  604009 like 201009 but u=604
  604010 like 201010 but u=604
  604100 like 201100 but mat=6041 rho=-6.234 u=604  $ ELEMENT F-4.1-1  S/N:5021
  604110 like 201110 but mat=6042 rho=-6.232 u=604  $ ELEMENT F-4.2-1  S/N:5021
  604120 like 201120 but mat=6043 rho=-6.230 u=604  $ ELEMENT F-4.3-1  S/N:5021
  604130 like 201130 but mat=6044 rho=-6.229 u=604  $ ELEMENT F-4.4-1  S/N:5021
  604140 like 201140 but mat=6043 rho=-6.230 u=604  $ ELEMENT F-4.5-1  S/N:5021
  604150 like 201150 but mat=6042 rho=-6.232 u=604  $ ELEMENT F-4.6-1  S/N:5021
  604160 like 201160 but mat=6041 rho=-6.234 u=604  $ ELEMENT F-4.7-1  S/N:5021
c ==============================================================================
  605000 like 201000 but u=605
  605001 like 201001 but u=605
  605002 like 201002 but u=605
  605003 like 201003 but u=605
  605004 like 201004 but u=605
  605005 like 201005 but u=605
  605006 like 201006 but u=605
  605007 like 201007 but u=605
  605008 like 201008 but u=605
  605009 like 201009 but u=605
  605010 like 201010 but u=605
  605100 like 201100 but mat=6051 rho=-6.234 u=605  $ ELEMENT F-5.1-1  S/N:5026
  605110 like 201110 but mat=6052 rho=-6.232 u=605  $ ELEMENT F-5.2-1  S/N:5026
  605120 like 201120 but mat=6053 rho=-6.230 u=605  $ ELEMENT F-5.3-1  S/N:5026
  605130 like 201130 but mat=6054 rho=-6.229 u=605  $ ELEMENT F-5.4-1  S/N:5026
  605140 like 201140 but mat=6053 rho=-6.230 u=605  $ ELEMENT F-5.5-1  S/N:5026
  605150 like 201150 but mat=6052 rho=-6.232 u=605  $ ELEMENT F-5.6-1  S/N:5026
  605160 like 201160 but mat=6051 rho=-6.234 u=605  $ ELEMENT F-5.7-1  S/N:5026
c ==============================================================================
  606000 like 201000 but u=606
  606001 like 201001 but u=606
  606002 like 201002 but u=606
  606003 like 201003 but u=606
  606004 like 201004 but u=606
  606005 like 201005 but u=606
  606006 like 201006 but u=606
  606007 like 201007 but u=606
  606008 like 201008 but u=606
  606009 like 201009 but u=606
  606010 like 201010 but u=606
  606100 like 201100 but mat=6061 rho=-6.405 u=606  $ ELEMENT F-6.1-1  S/N:6223
  606110 like 201110 but mat=6062 rho=-6.404 u=606  $ ELEMENT F-6.2-1  S/N:6223
  606120 like 201120 but mat=6063 rho=-6.404 u=606  $ ELEMENT F-6.3-1  S/N:6223
  606130 like 201130 but mat=6064 rho=-6.403 u=606  $ ELEMENT F-6.4-1  S/N:6223
  606140 like 201140 but mat=6063 rho=-6.404 u=606  $ ELEMENT F-6.5-1  S/N:6223
  606150 like 201150 but mat=6062 rho=-6.404 u=606  $ ELEMENT F-6.6-1  S/N:6223
  606160 like 201160 but mat=6061 rho=-6.405 u=606  $ ELEMENT F-6.7-1  S/N:6223
c ==============================================================================
  607000 like 201000 but u=607
  607001 like 201001 but u=607
  607002 like 201002 but u=607
  607003 like 201003 but u=607
  607004 like 201004 but u=607
  607005 like 201005 but u=607
  607006 like 201006 but u=607
  607007 like 201007 but u=607
  607008 like 201008 but u=607
  607009 like 201009 but u=607
  607010 like 201010 but u=607
  607100 like 201100 but mat=6071 rho=-6.233 u=607  $ ELEMENT F-7.1-1  S/N:5256
  607110 like 201110 but mat=6072 rho=-6.231 u=607  $ ELEMENT F-7.2-1  S/N:5256
  607120 like 201120 but mat=6073 rho=-6.229 u=607  $ ELEMENT F-7.3-1  S/N:5256
  607130 like 201130 but mat=6074 rho=-6.228 u=607  $ ELEMENT F-7.4-1  S/N:5256
  607140 like 201140 but mat=6073 rho=-6.229 u=607  $ ELEMENT F-7.5-1  S/N:5256
  607150 like 201150 but mat=6072 rho=-6.231 u=607  $ ELEMENT F-7.6-1  S/N:5256
  607160 like 201160 but mat=6071 rho=-6.233 u=607  $ ELEMENT F-7.7-1  S/N:5256
c ==============================================================================
  608000 like 201000 but u=608
  608001 like 201001 but u=608
  608002 like 201002 but u=608
  608003 like 201003 but u=608
  608004 like 201004 but u=608
  608005 like 201005 but u=608
  608006 like 201006 but u=608
  608007 like 201007 but u=608
  608008 like 201008 but u=608
  608009 like 201009 but u=608
  608010 like 201010 but u=608
  608100 like 201100 but mat=6081 rho=-6.240 u=608  $ ELEMENT F-8.1-1  S/N:6525
  608110 like 201110 but mat=6082 rho=-6.239 u=608  $ ELEMENT F-8.2-1  S/N:6525
  608120 like 201120 but mat=6083 rho=-6.239 u=608  $ ELEMENT F-8.3-1  S/N:6525
  608130 like 201130 but mat=6084 rho=-6.239 u=608  $ ELEMENT F-8.4-1  S/N:6525
  608140 like 201140 but mat=6083 rho=-6.239 u=608  $ ELEMENT F-8.5-1  S/N:6525
  608150 like 201150 but mat=6082 rho=-6.239 u=608  $ ELEMENT F-8.6-1  S/N:6525
  608160 like 201160 but mat=6081 rho=-6.240 u=608  $ ELEMENT F-8.7-1  S/N:6525
c ==============================================================================
  609000 like 201000 but u=609
  609001 like 201001 but u=609
  609002 like 201002 but u=609
  609003 like 201003 but u=609
  609004 like 201004 but u=609
  609005 like 201005 but u=609
  609006 like 201006 but u=609
  609007 like 201007 but u=609
  609008 like 201008 but u=609
  609009 like 201009 but u=609
  609010 like 201010 but u=609
  609100 like 201100 but mat=6091 rho=-6.235 u=609  $ ELEMENT F-9.1-1  S/N:5001
  609110 like 201110 but mat=6092 rho=-6.233 u=609  $ ELEMENT F-9.2-1  S/N:5001
  609120 like 201120 but mat=6093 rho=-6.231 u=609  $ ELEMENT F-9.3-1  S/N:5001
  609130 like 201130 but mat=6094 rho=-6.230 u=609  $ ELEMENT F-9.4-1  S/N:5001
  609140 like 201140 but mat=6093 rho=-6.231 u=609  $ ELEMENT F-9.5-1  S/N:5001
  609150 like 201150 but mat=6092 rho=-6.233 u=609  $ ELEMENT F-9.6-1  S/N:5001
  609160 like 201160 but mat=6091 rho=-6.235 u=609  $ ELEMENT F-9.7-1  S/N:5001
c ==============================================================================
  611000 like 201000 but u=611
  611001 like 201001 but u=611
  611002 like 201002 but u=611
  611003 like 201003 but u=611
  611004 like 201004 but u=611
  611005 like 201005 but u=611
  611006 like 201006 but u=611
  611007 like 201007 but u=611
  611008 like 201008 but u=611
  611009 like 201009 but u=611
  611010 like 201010 but u=611
  611100 like 201100 but mat=6111 rho=-6.232 u=611  $ ELEMENT F-11.1-1  S/N:5949
  611110 like 201110 but mat=6112 rho=-6.230 u=611  $ ELEMENT F-11.2-1  S/N:5949
  611120 like 201120 but mat=6113 rho=-6.228 u=611  $ ELEMENT F-11.3-1  S/N:5949
  611130 like 201130 but mat=6114 rho=-6.227 u=611  $ ELEMENT F-11.4-1  S/N:5949
  611140 like 201140 but mat=6113 rho=-6.228 u=611  $ ELEMENT F-11.5-1  S/N:5949
  611150 like 201150 but mat=6112 rho=-6.230 u=611  $ ELEMENT F-11.6-1  S/N:5949
  611160 like 201160 but mat=6111 rho=-6.232 u=611  $ ELEMENT F-11.7-1  S/N:5949
c ==============================================================================
  612000 like 201000 but u=612
  612001 like 201001 but u=612
  612002 like 201002 but u=612
  612003 like 201003 but u=612
  612004 like 201004 but u=612
  612005 like 201005 but u=612
  612006 like 201006 but u=612
  612007 like 201007 but u=612
  612008 like 201008 but u=612
  612009 like 201009 but u=612
  612010 like 201010 but u=612
  612100 like 201100 but mat=6121 rho=-6.502 u=612  $ ELEMENT F-12.1-1  S/N:2900
  612110 like 201110 but mat=6122 rho=-6.501 u=612  $ ELEMENT F-12.2-1  S/N:2900
  612120 like 201120 but mat=6123 rho=-6.501 u=612  $ ELEMENT F-12.3-1  S/N:2900
  612130 like 201130 but mat=6124 rho=-6.501 u=612  $ ELEMENT F-12.4-1  S/N:2900
  612140 like 201140 but mat=6123 rho=-6.501 u=612  $ ELEMENT F-12.5-1  S/N:2900
  612150 like 201150 but mat=6122 rho=-6.501 u=612  $ ELEMENT F-12.6-1  S/N:2900
  612160 like 201160 but mat=6121 rho=-6.502 u=612  $ ELEMENT F-12.7-1  S/N:2900
c ==============================================================================
  613000 like 201000 but u=613
  613001 like 201001 but u=613
  613002 like 201002 but u=613
  613003 like 201003 but u=613
  613004 like 201004 but u=613
  613005 like 201005 but u=613
  613006 like 201006 but u=613
  613007 like 201007 but u=613
  613008 like 201008 but u=613
  613009 like 201009 but u=613
  613010 like 201010 but u=613
  613100 like 201100 but mat=6131 rho=-6.072 u=613  $ ELEMENT F-13.1-1  S/N:3329
  613110 like 201110 but mat=6132 rho=-6.071 u=613  $ ELEMENT F-13.2-1  S/N:3329
  613120 like 201120 but mat=6133 rho=-6.070 u=613  $ ELEMENT F-13.3-1  S/N:3329
  613130 like 201130 but mat=6134 rho=-6.069 u=613  $ ELEMENT F-13.4-1  S/N:3329
  613140 like 201140 but mat=6133 rho=-6.070 u=613  $ ELEMENT F-13.5-1  S/N:3329
  613150 like 201150 but mat=6132 rho=-6.071 u=613  $ ELEMENT F-13.6-1  S/N:3329
  613160 like 201160 but mat=6131 rho=-6.072 u=613  $ ELEMENT F-13.7-1  S/N:3329
c ==============================================================================
  614000 like 201000 but u=614
  614001 like 201001 but u=614
  614002 like 201002 but u=614
  614003 like 201003 but u=614
  614004 like 201004 but u=614
  614005 like 201005 but u=614
  614006 like 201006 but u=614
  614007 like 201007 but u=614
  614008 like 201008 but u=614
  614009 like 201009 but u=614
  614010 like 201010 but u=614
  614100 like 201100 but mat=6141 rho=-6.232 u=614  $ ELEMENT F-14.1-1  S/N:5653
  614110 like 201110 but mat=6142 rho=-6.230 u=614  $ ELEMENT F-14.2-1  S/N:5653
  614120 like 201120 but mat=6143 rho=-6.228 u=614  $ ELEMENT F-14.3-1  S/N:5653
  614130 like 201130 but mat=6144 rho=-6.227 u=614  $ ELEMENT F-14.4-1  S/N:5653
  614140 like 201140 but mat=6143 rho=-6.228 u=614  $ ELEMENT F-14.5-1  S/N:5653
  614150 like 201150 but mat=6142 rho=-6.230 u=614  $ ELEMENT F-14.6-1  S/N:5653
  614160 like 201160 but mat=6141 rho=-6.232 u=614  $ ELEMENT F-14.7-1  S/N:5653
c ==============================================================================
  615000 like 201000 but u=615
  615001 like 201001 but u=615
  615002 like 201002 but u=615
  615003 like 201003 but u=615
  615004 like 201004 but u=615
  615005 like 201005 but u=615
  615006 like 201006 but u=615
  615007 like 201007 but u=615
  615008 like 201008 but u=615
  615009 like 201009 but u=615
  615010 like 201010 but u=615
  615100 like 201100 but mat=6151 rho=-5.939 u=615  $ ELEMENT F-15.1-1  S/N:2448
  615110 like 201110 but mat=6152 rho=-5.938 u=615  $ ELEMENT F-15.2-1  S/N:2448
  615120 like 201120 but mat=6153 rho=-5.937 u=615  $ ELEMENT F-15.3-1  S/N:2448
  615130 like 201130 but mat=6154 rho=-5.936 u=615  $ ELEMENT F-15.4-1  S/N:2448
  615140 like 201140 but mat=6153 rho=-5.937 u=615  $ ELEMENT F-15.5-1  S/N:2448
  615150 like 201150 but mat=6152 rho=-5.938 u=615  $ ELEMENT F-15.6-1  S/N:2448
  615160 like 201160 but mat=6151 rho=-5.939 u=615  $ ELEMENT F-15.7-1  S/N:2448
c ==============================================================================
  616000 like 201000 but u=616
  616001 like 201001 but u=616
  616002 like 201002 but u=616
  616003 like 201003 but u=616
  616004 like 201004 but u=616
  616005 like 201005 but u=616
  616006 like 201006 but u=616
  616007 like 201007 but u=616
  616008 like 201008 but u=616
  616009 like 201009 but u=616
  616010 like 201010 but u=616
  616100 like 201100 but mat=6161 rho=-6.232 u=616  $ ELEMENT F-16.1-1  S/N:5655
  616110 like 201110 but mat=6162 rho=-6.230 u=616  $ ELEMENT F-16.2-1  S/N:5655
  616120 like 201120 but mat=6163 rho=-6.228 u=616  $ ELEMENT F-16.3-1  S/N:5655
  616130 like 201130 but mat=6164 rho=-6.227 u=616  $ ELEMENT F-16.4-1  S/N:5655
  616140 like 201140 but mat=6163 rho=-6.228 u=616  $ ELEMENT F-16.5-1  S/N:5655
  616150 like 201150 but mat=6162 rho=-6.230 u=616  $ ELEMENT F-16.6-1  S/N:5655
  616160 like 201160 but mat=6161 rho=-6.232 u=616  $ ELEMENT F-16.7-1  S/N:5655
c ==============================================================================
  617000 like 201000 but u=617
  617001 like 201001 but u=617
  617002 like 201002 but u=617
  617003 like 201003 but u=617
  617004 like 201004 but u=617
  617005 like 201005 but u=617
  617006 like 201006 but u=617
  617007 like 201007 but u=617
  617008 like 201008 but u=617
  617009 like 201009 but u=617
  617010 like 201010 but u=617
  617100 like 201100 but mat=6171 rho=-6.232 u=617  $ ELEMENT F-17.1-1  S/N:5939
  617110 like 201110 but mat=6172 rho=-6.229 u=617  $ ELEMENT F-17.2-1  S/N:5939
  617120 like 201120 but mat=6173 rho=-6.227 u=617  $ ELEMENT F-17.3-1  S/N:5939
  617130 like 201130 but mat=6174 rho=-6.226 u=617  $ ELEMENT F-17.4-1  S/N:5939
  617140 like 201140 but mat=6173 rho=-6.227 u=617  $ ELEMENT F-17.5-1  S/N:5939
  617150 like 201150 but mat=6172 rho=-6.229 u=617  $ ELEMENT F-17.6-1  S/N:5939
  617160 like 201160 but mat=6171 rho=-6.232 u=617  $ ELEMENT F-17.7-1  S/N:5939
c ==============================================================================
  618000 like 201000 but u=618
  618001 like 201001 but u=618
  618002 like 201002 but u=618
  618003 like 201003 but u=618
  618004 like 201004 but u=618
  618005 like 201005 but u=618
  618006 like 201006 but u=618
  618007 like 201007 but u=618
  618008 like 201008 but u=618
  618009 like 201009 but u=618
  618010 like 201010 but u=618
  618100 like 201100 but mat=6181 rho=-6.232 u=618  $ ELEMENT F-18.1-1  S/N:5946
  618110 like 201110 but mat=6182 rho=-6.230 u=618  $ ELEMENT F-18.2-1  S/N:5946
  618120 like 201120 but mat=6183 rho=-6.227 u=618  $ ELEMENT F-18.3-1  S/N:5946
  618130 like 201130 but mat=6184 rho=-6.227 u=618  $ ELEMENT F-18.4-1  S/N:5946
  618140 like 201140 but mat=6183 rho=-6.227 u=618  $ ELEMENT F-18.5-1  S/N:5946
  618150 like 201150 but mat=6182 rho=-6.230 u=618  $ ELEMENT F-18.6-1  S/N:5946
  618160 like 201160 but mat=6181 rho=-6.232 u=618  $ ELEMENT F-18.7-1  S/N:5946
c ==============================================================================
  619000 like 201000 but u=619
  619001 like 201001 but u=619
  619002 like 201002 but u=619
  619003 like 201003 but u=619
  619004 like 201004 but u=619
  619005 like 201005 but u=619
  619006 like 201006 but u=619
  619007 like 201007 but u=619
  619008 like 201008 but u=619
  619009 like 201009 but u=619
  619010 like 201010 but u=619
  619100 like 201100 but mat=6191 rho=-6.073 u=619  $ ELEMENT F-19.1-1  S/N:3113
  619110 like 201110 but mat=6192 rho=-6.072 u=619  $ ELEMENT F-19.2-1  S/N:3113
  619120 like 201120 but mat=6193 rho=-6.070 u=619  $ ELEMENT F-19.3-1  S/N:3113
  619130 like 201130 but mat=6194 rho=-6.070 u=619  $ ELEMENT F-19.4-1  S/N:3113
  619140 like 201140 but mat=6193 rho=-6.070 u=619  $ ELEMENT F-19.5-1  S/N:3113
  619150 like 201150 but mat=6192 rho=-6.072 u=619  $ ELEMENT F-19.6-1  S/N:3113
  619160 like 201160 but mat=6191 rho=-6.073 u=619  $ ELEMENT F-19.7-1  S/N:3113
c ==============================================================================
  620000 like 201000 but u=620
  620001 like 201001 but u=620
  620002 like 201002 but u=620
  620003 like 201003 but u=620
  620004 like 201004 but u=620
  620005 like 201005 but u=620
  620006 like 201006 but u=620
  620007 like 201007 but u=620
  620008 like 201008 but u=620
  620009 like 201009 but u=620
  620010 like 201010 but u=620
  620100 like 201100 but mat=6201 rho=-6.307 u=620  $ ELEMENT F-20.1-1  S/N:2949
  620110 like 201110 but mat=6202 rho=-6.306 u=620  $ ELEMENT F-20.2-1  S/N:2949
  620120 like 201120 but mat=6203 rho=-6.306 u=620  $ ELEMENT F-20.3-1  S/N:2949
  620130 like 201130 but mat=6204 rho=-6.306 u=620  $ ELEMENT F-20.4-1  S/N:2949
  620140 like 201140 but mat=6203 rho=-6.306 u=620  $ ELEMENT F-20.5-1  S/N:2949
  620150 like 201150 but mat=6202 rho=-6.306 u=620  $ ELEMENT F-20.6-1  S/N:2949
  620160 like 201160 but mat=6201 rho=-6.307 u=620  $ ELEMENT F-20.7-1  S/N:2949
c ==============================================================================
  621000 like 201000 but u=621
  621001 like 201001 but u=621
  621002 like 201002 but u=621
  621003 like 201003 but u=621
  621004 like 201004 but u=621
  621005 like 201005 but u=621
  621006 like 201006 but u=621
  621007 like 201007 but u=621
  621008 like 201008 but u=621
  621009 like 201009 but u=621
  621010 like 201010 but u=621
  621100 like 201100 but mat=6211 rho=-6.234 u=621  $ ELEMENT F-21.1-1  S/N:5649
  621110 like 201110 but mat=6212 rho=-6.232 u=621  $ ELEMENT F-21.2-1  S/N:5649
  621120 like 201120 but mat=6213 rho=-6.230 u=621  $ ELEMENT F-21.3-1  S/N:5649
  621130 like 201130 but mat=6214 rho=-6.229 u=621  $ ELEMENT F-21.4-1  S/N:5649
  621140 like 201140 but mat=6213 rho=-6.230 u=621  $ ELEMENT F-21.5-1  S/N:5649
  621150 like 201150 but mat=6212 rho=-6.232 u=621  $ ELEMENT F-21.6-1  S/N:5649
  621160 like 201160 but mat=6211 rho=-6.234 u=621  $ ELEMENT F-21.7-1  S/N:5649
c ==============================================================================
  622000 like 201000 but u=622
  622001 like 201001 but u=622
  622002 like 201002 but u=622
  622003 like 201003 but u=622
  622004 like 201004 but u=622
  622005 like 201005 but u=622
  622006 like 201006 but u=622
  622007 like 201007 but u=622
  622008 like 201008 but u=622
  622009 like 201009 but u=622
  622010 like 201010 but u=622
  622100 like 201100 but mat=6221 rho=-6.031 u=622  $ ELEMENT F-22.1-1  S/N:2917
  622110 like 201110 but mat=6222 rho=-6.030 u=622  $ ELEMENT F-22.2-1  S/N:2917
  622120 like 201120 but mat=6223 rho=-6.030 u=622  $ ELEMENT F-22.3-1  S/N:2917
  622130 like 201130 but mat=6224 rho=-6.030 u=622  $ ELEMENT F-22.4-1  S/N:2917
  622140 like 201140 but mat=6223 rho=-6.030 u=622  $ ELEMENT F-22.5-1  S/N:2917
  622150 like 201150 but mat=6222 rho=-6.030 u=622  $ ELEMENT F-22.6-1  S/N:2917
  622160 like 201160 but mat=6221 rho=-6.031 u=622  $ ELEMENT F-22.7-1  S/N:2917
c ==============================================================================
  623000 like 201000 but u=623
  623001 like 201001 but u=623
  623002 like 201002 but u=623
  623003 like 201003 but u=623
  623004 like 201004 but u=623
  623005 like 201005 but u=623
  623006 like 201006 but u=623
  623007 like 201007 but u=623
  623008 like 201008 but u=623
  623009 like 201009 but u=623
  623010 like 201010 but u=623
  623100 like 201100 but mat=6231 rho=-6.232 u=623  $ ELEMENT F-23.1-1  S/N:5000
  623110 like 201110 but mat=6232 rho=-6.230 u=623  $ ELEMENT F-23.2-1  S/N:5000
  623120 like 201120 but mat=6233 rho=-6.228 u=623  $ ELEMENT F-23.3-1  S/N:5000
  623130 like 201130 but mat=6234 rho=-6.227 u=623  $ ELEMENT F-23.4-1  S/N:5000
  623140 like 201140 but mat=6233 rho=-6.228 u=623  $ ELEMENT F-23.5-1  S/N:5000
  623150 like 201150 but mat=6232 rho=-6.230 u=623  $ ELEMENT F-23.6-1  S/N:5000
  623160 like 201160 but mat=6231 rho=-6.232 u=623  $ ELEMENT F-23.7-1  S/N:5000
c ==============================================================================
  624000 like 201000 but u=624
  624001 like 201001 but u=624
  624002 like 201002 but u=624
  624003 like 201003 but u=624
  624004 like 201004 but u=624
  624005 like 201005 but u=624
  624006 like 201006 but u=624
  624007 like 201007 but u=624
  624008 like 201008 but u=624
  624009 like 201009 but u=624
  624010 like 201010 but u=624
  624100 like 201100 but mat=6241 rho=-6.232 u=624  $ ELEMENT F-24.1-1  S/N:5948
  624110 like 201110 but mat=6242 rho=-6.229 u=624  $ ELEMENT F-24.2-1  S/N:5948
  624120 like 201120 but mat=6243 rho=-6.227 u=624  $ ELEMENT F-24.3-1  S/N:5948
  624130 like 201130 but mat=6244 rho=-6.226 u=624  $ ELEMENT F-24.4-1  S/N:5948
  624140 like 201140 but mat=6243 rho=-6.227 u=624  $ ELEMENT F-24.5-1  S/N:5948
  624150 like 201150 but mat=6242 rho=-6.229 u=624  $ ELEMENT F-24.6-1  S/N:5948
  624160 like 201160 but mat=6241 rho=-6.232 u=624  $ ELEMENT F-24.7-1  S/N:5948
c ==============================================================================
  625000 like 201000 but u=625
  625001 like 201001 but u=625
  625002 like 201002 but u=625
  625003 like 201003 but u=625
  625004 like 201004 but u=625
  625005 like 201005 but u=625
  625006 like 201006 but u=625
  625007 like 201007 but u=625
  625008 like 201008 but u=625
  625009 like 201009 but u=625
  625010 like 201010 but u=625
  625100 like 201100 but mat=6251 rho=-6.233 u=625  $ ELEMENT F-25.1-1  S/N:5022
  625110 like 201110 but mat=6252 rho=-6.230 u=625  $ ELEMENT F-25.2-1  S/N:5022
  625120 like 201120 but mat=6253 rho=-6.228 u=625  $ ELEMENT F-25.3-1  S/N:5022
  625130 like 201130 but mat=6254 rho=-6.227 u=625  $ ELEMENT F-25.4-1  S/N:5022
  625140 like 201140 but mat=6253 rho=-6.228 u=625  $ ELEMENT F-25.5-1  S/N:5022
  625150 like 201150 but mat=6252 rho=-6.230 u=625  $ ELEMENT F-25.6-1  S/N:5022
  625160 like 201160 but mat=6251 rho=-6.233 u=625  $ ELEMENT F-25.7-1  S/N:5022
c ==============================================================================
  626000 like 201000 but u=626
  626001 like 201001 but u=626
  626002 like 201002 but u=626
  626003 like 201003 but u=626
  626004 like 201004 but u=626
  626005 like 201005 but u=626
  626006 like 201006 but u=626
  626007 like 201007 but u=626
  626008 like 201008 but u=626
  626009 like 201009 but u=626
  626010 like 201010 but u=626
  626100 like 201100 but mat=6261 rho=-6.495 u=626  $ ELEMENT F-26.1-1  S/N:2907
  626110 like 201110 but mat=6262 rho=-6.495 u=626  $ ELEMENT F-26.2-1  S/N:2907
  626120 like 201120 but mat=6263 rho=-6.494 u=626  $ ELEMENT F-26.3-1  S/N:2907
  626130 like 201130 but mat=6264 rho=-6.494 u=626  $ ELEMENT F-26.4-1  S/N:2907
  626140 like 201140 but mat=6263 rho=-6.494 u=626  $ ELEMENT F-26.5-1  S/N:2907
  626150 like 201150 but mat=6262 rho=-6.495 u=626  $ ELEMENT F-26.6-1  S/N:2907
  626160 like 201160 but mat=6261 rho=-6.495 u=626  $ ELEMENT F-26.7-1  S/N:2907
c ==============================================================================
  627000 like 201000 but u=627
  627001 like 201001 but u=627
  627002 like 201002 but u=627
  627003 like 201003 but u=627
  627004 like 201004 but u=627
  627005 like 201005 but u=627
  627006 like 201006 but u=627
  627007 like 201007 but u=627
  627008 like 201008 but u=627
  627009 like 201009 but u=627
  627010 like 201010 but u=627
  627100 like 201100 but mat=6271 rho=-6.396 u=627  $ ELEMENT F-27.1-1  S/N:5944
  627110 like 201110 but mat=6272 rho=-6.394 u=627  $ ELEMENT F-27.2-1  S/N:5944
  627120 like 201120 but mat=6273 rho=-6.391 u=627  $ ELEMENT F-27.3-1  S/N:5944
  627130 like 201130 but mat=6274 rho=-6.391 u=627  $ ELEMENT F-27.4-1  S/N:5944
  627140 like 201140 but mat=6273 rho=-6.391 u=627  $ ELEMENT F-27.5-1  S/N:5944
  627150 like 201150 but mat=6272 rho=-6.394 u=627  $ ELEMENT F-27.6-1  S/N:5944
  627160 like 201160 but mat=6271 rho=-6.396 u=627  $ ELEMENT F-27.7-1  S/N:5944
c ==============================================================================
  628000 like 201000 but u=628
  628001 like 201001 but u=628
  628002 like 201002 but u=628
  628003 like 201003 but u=628
  628004 like 201004 but u=628
  628005 like 201005 but u=628
  628006 like 201006 but u=628
  628007 like 201007 but u=628
  628008 like 201008 but u=628
  628009 like 201009 but u=628
  628010 like 201010 but u=628
  628100 like 201100 but mat=6281 rho=-6.236 u=628  $ ELEMENT F-28.1-1  S/N:3326
  628110 like 201110 but mat=6282 rho=-6.235 u=628  $ ELEMENT F-28.2-1  S/N:3326
  628120 like 201120 but mat=6283 rho=-6.233 u=628  $ ELEMENT F-28.3-1  S/N:3326
  628130 like 201130 but mat=6284 rho=-6.233 u=628  $ ELEMENT F-28.4-1  S/N:3326
  628140 like 201140 but mat=6283 rho=-6.233 u=628  $ ELEMENT F-28.5-1  S/N:3326
  628150 like 201150 but mat=6282 rho=-6.235 u=628  $ ELEMENT F-28.6-1  S/N:3326
  628160 like 201160 but mat=6281 rho=-6.236 u=628  $ ELEMENT F-28.7-1  S/N:3326
c ==============================================================================
  629000 like 201000 but u=629
  629001 like 201001 but u=629
  629002 like 201002 but u=629
  629003 like 201003 but u=629
  629004 like 201004 but u=629
  629005 like 201005 but u=629
  629006 like 201006 but u=629
  629007 like 201007 but u=629
  629008 like 201008 but u=629
  629009 like 201009 but u=629
  629010 like 201010 but u=629
  629100 like 201100 but mat=6291 rho=-6.273 u=629  $ ELEMENT F-29.1-1  S/N:2914
  629110 like 201110 but mat=6292 rho=-6.273 u=629  $ ELEMENT F-29.2-1  S/N:2914
  629120 like 201120 but mat=6293 rho=-6.272 u=629  $ ELEMENT F-29.3-1  S/N:2914
  629130 like 201130 but mat=6294 rho=-6.272 u=629  $ ELEMENT F-29.4-1  S/N:2914
  629140 like 201140 but mat=6293 rho=-6.272 u=629  $ ELEMENT F-29.5-1  S/N:2914
  629150 like 201150 but mat=6292 rho=-6.273 u=629  $ ELEMENT F-29.6-1  S/N:2914
  629160 like 201160 but mat=6291 rho=-6.273 u=629  $ ELEMENT F-29.7-1  S/N:2914
c ==============================================================================
  630000 like 201000 but u=630
  630001 like 201001 but u=630
  630002 like 201002 but u=630
  630003 like 201003 but u=630
  630004 like 201004 but u=630
  630005 like 201005 but u=630
  630006 like 201006 but u=630
  630007 like 201007 but u=630
  630008 like 201008 but u=630
  630009 like 201009 but u=630
  630010 like 201010 but u=630
  630100 like 201100 but mat=6301 rho=-6.177 u=630  $ ELEMENT F-30.1-1  S/N:2909
  630110 like 201110 but mat=6302 rho=-6.177 u=630  $ ELEMENT F-30.2-1  S/N:2909
  630120 like 201120 but mat=6303 rho=-6.176 u=630  $ ELEMENT F-30.3-1  S/N:2909
  630130 like 201130 but mat=6304 rho=-6.176 u=630  $ ELEMENT F-30.4-1  S/N:2909
  630140 like 201140 but mat=6303 rho=-6.176 u=630  $ ELEMENT F-30.5-1  S/N:2909
  630150 like 201150 but mat=6302 rho=-6.177 u=630  $ ELEMENT F-30.6-1  S/N:2909
  630160 like 201160 but mat=6301 rho=-6.177 u=630  $ ELEMENT F-30.7-1  S/N:2909
c ==============================================================================
c ** RABBIT: UNIVERSE = [2]
  145     7  -2.699   (25 -12):(12 219):(-25 -45) u=2 $ Not tube
  146     3  -0.001239 12 -219                    u=2 $ air
  147     6  -1        #145 #146                  u=2 $water
c ** GRAPHITE ROD: UNIVERSE = [3]
  160     6      -1 -21 201              u=3 $ Water around lower pin
  161     9    -7.9 -21 -201             u=3 $ Bottom pin
  162     9    -7.9 -14 21               u=3 $ BOTTOM SS cap
  163     9    -7.9 14 -4 202            u=3 $ Cladding
  164     2    -1.6 14 -4 -202           u=3 $ GRAPHITE
  165     9    -7.9 4 -3                 u=3 $ TOP SS cap
  166     9    -7.9 3 -201               u=3 $ Upper pin
  167     6      -1 3 201                u=3 $ Water around pin
c ** CENTRAL THIMBLE: UNIVERSE = [4]
  170     6      -1 12 -202              u=4 $ Water over midplane
  171     7  -2.699 -12 :202             u=4 $ Below centerline
c ** SOURCE: UNIVERSE = [5]
  175     7  -2.699 -233 :230 :232       u=5 $ SOURCE HOLDER
  176     9 -0.394473 233 -234 -231      u=5 $ SOURCE, SS SHELL
  177     3 -0.001239 -232 233 -230 #176 u=5 $ SOURCE CAVITY
c ** THIMBLE: UNIVERSE = [6]
  180     6      -1 -21 201              u=6 $ Water around lower pin
  181     7  -2.699 -21 -201             u=6 $ Bottom pin
  182     7  -2.699 21 -14               u=6 $ BOTTOM cap
  183     7  -2.699 14 210               u=6 $ Thimble TUBE
  184     6      -1 14 -210              u=6 $ Thimble Water
c ** CORE & FUEL: UNIVERSE = [7]
c ** WATER AROUND UPPER PINS & GRID PLATE ***********************************
  185     6      -1 -1 5 1201 1202 1203 1204 1205 1206 1301 1302 1303 $ H20
             1304 1305 1306 1307 1308 1309 1310 1311 1312 1401 1402 1403 1404
             1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415 1416 1417
             1418 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510 1511 1512
             1513 1514 1515 1516 1517 1518 1519 1520 1521 1522 1523 1524 1601
             1602 1603 1604 1605 1606 1607 1608 1609 1610 1611 1612 1613 1614
             1615 1616 1617 1618 1619 1620 1621 1622 1623 1624 1625 1626 1627
             1628 1629 1630 203                                      u=7
c ** UPPER GRID PLATE *******************************************************
  186     7  -2.699 -5 8 1201 1202 1203 1204 1205 1206 1301 $ U17ER GRID PLAT
             1302 1303 1304 1305 1306 1307 1308 1309 1310 1311 1312 1401 1402
             1403 1404 1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415
             1416 1417 1418 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510
             1511 1512 1513 1514 1515 1516 1517 1518 1519 1520 1521 1522 1523
             1524 1601 1602 1603 1604 1605 1606 1607 1608 1609 1610 1611 1612
             1613 1614 1615 1616 1617 1618 1619 1620 1621 1622 1623 1624 1625
             1626 1627 1628 1629 1630 203                            u=7
c ** CORE WATER *************************************************************
  187     6      -1 -8 1201 1202 1203 1204 1205 1206 1301 1302 1303 $ CORE H2O
             1304 1305 1306 1307 1308 1309 1310 1311 1312 1401 1402 1403 1404
             1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415 1416 1417
             1418 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510 1511 1512
             1513 1514 1515 1516 1517 1518 1519 1520 1521 1522 1523 1524 1601
             1602 1603 1604 1605 1606 1607 1608 1609 1610 1611 1612 1613 1614
             1615 1616 1617 1618 1619 1620 1621 1622 1623 1624 1625 1626 1627
             1628 1629 1630 203                                      u=7
c ** FUEL POSITIONS *********************************************************
 1100     0         -203   fill=4   (  0.000000   0.000000  0.00)  u=7 $ A1
  201     0         -1201  fill=201 (3.121883  2.586013  0.00)  u=7 $ B1
  202     0         -1202  fill=202 (3.800495  -1.410623  0.00)  u=7 $ B2
  203     0         -1203  fill=203 (0.678612  -3.996636  0.00)  u=7 $ B3
  204     0         -1204  fill=204 (-3.121883  -2.586013  0.00)  u=7 $ B4
  205     0         -1205  fill=205 (-3.800495  1.410623  0.00)  u=7 $ B5
  206     0         -1206  fill=206 (-0.678612  3.996636  0.00)  u=7 $ B6
  301     0         -1301  fill=301 (6.145963  5.091011  0.00)  u=7 $ C1
  302     0         -1302  fill=302 (7.868066  1.335964  0.00)  u=7 $ C2
  303     0         -1303  fill=303 (7.481927  -2.777054  0.00)  u=7 $ C3
  304     0         -1304  fill=304 (5.091011  -6.145963  0.00)  u=7 $ C4
  305     0         -1305  fill=305 (1.335964  -7.868066  0.00)  u=7 $ C5
  306     0         -1306  fill=306 (-2.777054  -7.481927  0.00)  u=7 $ C6
  307     0         -1307  fill=21  (-6.145963  -5.091011  39.37)  u=7 $ C7
  308     0         -1308  fill=308 (-7.868066  -1.335964  0.00)  u=7 $ C8
  309     0         -1309  fill=309 (-7.481927  2.777054  0.00)  u=7 $ C9
  310     0         -1310  fill=310 (-5.091011  6.145963  0.00)  u=7 $ C10
  311     0         -1311  fill=311 (-1.335964  7.868066  0.00)  u=7 $ C11
  312     0         -1312  fill=312 (2.777054  7.481927  0.00)  u=7 $ C12
  401     0         -1401  fill=401 (9.199383  7.620314  0.00)  u=7 $ D1
  402     0         -1402  fill=402 (11.250894  4.014378  0.00)  u=7 $ D2
  403     0         -1403  fill=403 (11.945380  -0.075751  0.00)  u=7 $ D3
  404     0         -1404  fill=22  (11.199077  -4.156742  3.41)  u=7 $ D4
  405     0         -1405  fill=405 (9.102000  -7.736370  0.00)  u=7 $ D5
  406     0         -1406  fill=406 (5.907088  -10.382877  0.00)  u=7 $ D6
  407     0         -1407  fill=407 (1.999694  -11.777056  0.00)  u=7 $ D7
  408     0         -1408  fill=408 (-2.148893  -11.750749  0.00)  u=7 $ D8
  409     0         -1409  fill=409 (-6.038292  -10.307127  0.00)  u=7 $ D9
  410     0         -1410  fill=410 (-9.199383  -7.620314  0.00)  u=7 $ D10
  411     0         -1411  fill=411 (-11.250894  -4.014378  0.00)  u=7 $ D11
  412     0         -1412  fill=412 (-11.945380  0.075751  0.00)  u=7 $ D12
  413     0         -1413  fill=413 (-11.199077  4.156742  0.00)  u=7 $ D13
  414     0         -1414  fill=414 (-9.102000  7.736370  0.00)  u=7 $ D14
  415     0         -1415  fill=415 (-5.907088  10.382877  0.00)  u=7 $ D15
  416     0         -1416  fill=23  (-1.999694  11.777056  20.28)  u=7 $ D16
  417     0         -1417  fill=417 (2.148893  11.750749  0.00)  u=7 $ D17
  418     0         -1418  fill=418 (6.038292  10.307127  0.00)  u=7 $ D18
  501     0         -1501  fill=24  (12.256716  10.152857  12.45)  u=7 $ E1
  502     0         -1502  fill=502 (14.466832  6.634635  0.00)  u=7 $ E2
  503     0         -1503  fill=503 (15.691056  2.664274  0.00)  u=7 $ E3
  504     0         -1504  fill=504 (15.845961  -1.487653  0.00)  u=7 $ E4
  505     0         -1505  fill=505 (14.920990  -5.538199  0.00)  u=7 $ E5
  506     0         -1506  fill=506 (12.979179  -9.211326  0.00)  u=7 $ E6
  507     0         -1507  fill=507 (10.152857  -12.256716  0.00)  u=7 $ E7
  508     0         -1508  fill=508 (6.634635  -14.466832  0.00)  u=7 $ E8
  509     0         -1509  fill=509 (2.664274  -15.691056  0.00)  u=7 $ E9
  510     0         -1510  fill=510 (-1.487653  -15.845961  0.00)  u=7 $ E10
  511     0         -1511  fill=511 (-5.538199  -14.920990  0.00)  u=7 $ E11
  512     0         -1512  fill=512 (-9.211326  -12.979179  0.00)  u=7 $ E12
  513     0         -1513  fill=513 (-12.256716  -10.152857  0.00)  u=7 $ E13
  514     0         -1514  fill=514 (-14.466832  -6.634635  0.00)  u=7 $ E14
  515     0         -1515  fill=515 (-15.691056  -2.664274  0.00)  u=7 $ E15
  516     0         -1516  fill=516 (-15.845961  1.487653  0.00)  u=7 $ E16
  517     0         -1517  fill=517 (-14.920990  5.538199  0.00)  u=7 $ E17
  518     0         -1518  fill=518 (-12.979179  9.211326  0.00)  u=7 $ E18
  519     0         -1519  fill=519 (-10.152857  12.256716  0.00)  u=7 $ E19
  520     0         -1520  fill=520 (-6.634635  14.466832  0.00)  u=7 $ E20
  521     0         -1521  fill=521 (-2.664274  15.691056  0.00)  u=7 $ E21
  522     0         -1522  fill=522 (1.487653  15.845961  0.00)  u=7 $ E22
  523     0         -1523  fill=523 (5.538199  14.920990  0.00)  u=7 $ E23
  524     0         -1524  fill=524 (9.211326  12.979179  0.00)  u=7 $ E24
  601     0         -1601  fill=601 (15.316005  12.687021  0.00)  u=7 $ F1
  602     0         -1602  fill=602 (17.619093  9.225402  0.00)  u=7 $ F2
  603     0         -1603  fill=603 (19.152143  5.360590  0.00)  u=7 $ F3
  604     0         -1604  fill=604 (19.848152  1.261493  0.00)  u=7 $ F4
  605     0         -1605  fill=605 (19.676701  -2.892736  0.00)  u=7 $ F5
  606     0         -1606  fill=606 (18.645284  -6.920539  0.00)  u=7 $ F6
  607     0         -1607  fill=607 (16.798979  -10.645881  0.00)  u=7 $ F7
  608     0         -1608  fill=608 (14.218478  -13.905947  0.00)  u=7 $ F8
  609     0         -1609  fill=609 (11.016561  -16.558257  0.00)  u=7 $ F9
  610     0         -1610  fill=5   (7.333167  -18.486891  0.00)  u=7 $ F10
  611     0         -1611  fill=611 (3.329279  -19.607559  0.00)  u=7 $ F11
  612     0         -1612  fill=612 (-0.820114  -19.871283  0.00)  u=7 $ F12
  613     0         -1613  fill=613 (-4.933665  -19.266538  0.00)  u=7 $ F13
  614     0         -1614  fill=614 (-8.831591  -17.819751  0.00)  u=7 $ F14
  615     0         -1615  fill=615 (-12.343534  -15.594155  0.00)  u=7 $ F15
  616     0         -1616  fill=616 (-15.316005  -12.687021  0.00)  u=7 $ F16
  617     0         -1617  fill=617 (-17.619093  -9.225402  0.00)  u=7 $ F17
  618     0         -1618  fill=618 (-19.152143  -5.360590  0.00)  u=7 $ F18
  619     0         -1619  fill=619 (-19.848152  -1.261493  0.00)  u=7 $ F19
  620     0         -1620  fill=620 (-19.676701  2.892736  0.00)  u=7 $ F20
  621     0         -1621  fill=621 (-18.645284  6.920539  0.00)  u=7 $ F21
  622     0         -1622  fill=622 (-16.798979  10.645881  0.00)  u=7 $ F22
  623     0         -1623  fill=623 (-14.218478  13.905947  0.00)  u=7 $ F23
  624     0         -1624  fill=624 (-11.016561  16.558257  0.00)  u=7 $ F24
  625     0         -1625  fill=625 (-7.333167  18.486891  0.00)  u=7 $ F25
  626     0         -1626  fill=626 (-3.329279  19.607559  0.00)  u=7 $ F26
  627     0         -1627  fill=627 (0.820114  19.871283  0.00)  u=7 $ F27
  628     0         -1628  fill=628 (4.933665  19.266538  0.00)  u=7 $ F28
  629     0         -1629  fill=629 (8.831591  17.819751  0.00)  u=7 $ F29
  630     0         -1630  fill=630 (12.343534  15.594155  0.00)  u=7 $ F30

c SURFACES
c ***  ELEVATIONS **************************************************************
    1        pz 50  $ POOL SURFACE
    2        pz 37.2618  $ TOP OF FUEL PINS
    3        pz 33.782  $ Top of end cap
    4        pz 30.3276  $ Top of gap/CLADDING
    5        pz 33.79  $ Top of grid plate
    6        pz 27.7876  $ Top of axial reflector
    7        pz 27.94  $ REFLECTOR CANNISTER
    8        pz 31.162  $ **** Bottom top grid
    9        pz 26.035  $ RADIAL REFLECTOR TOP
   10        pz 19.05  $ Top of fuel
   11        pz 0.635  $ RSR FLOOR
   12        pz 0  $ RSR SUBFLOOR -- LINE OF FUEL SYMMETRTY
   13        pz -19.05  $ Bottom of fuel (38.1/2 cm)
c **  BEST ESTIMATE 3.74 in = 9.4996 cm {-19.05 cm -9.4996 cm}=>
   14        pz -27.7876 $ Bottom of axial reflector/CLADDING
c **  APPROX CONE 3.797 cm base 3.797 cm heigth
   21        pz -32      $ Bottom of plug/top of pin
   15        pz -34.163  $ Top of bottom grid plate
   16        pz -26.035  $ RADIAL REFLECTOR BOTTOM
   17        pz -27.305  $ REFLECTOR CANNISTER BOTTOM
   18        pz -36.07  $ Bottom of grid plate
   20        pz -50  $ BOTTOM OF POOL
c *** OTHER PLANES *************************************************************
   25        pz -5.0 $ Bottom of thick part of rabbit
   26        px 0  $ CORE 1/2
  100        pz 13.6071428571  $ FUEL division
  101        pz 8.16428571429  $ FUEL division
  102        pz 2.72142857143  $ FUEL division
  103        pz -2.72142857143  $ FUEL division
  104        pz -8.16428571429  $ FUEL division
  105        pz -13.6071428571  $ FUEL division
  111        px  23.750          $ PLUG 1               !!!
  112        px  33.91           $ PLUG 2               !!!
  113        px  125.35          $ PLUG 3               !!!
  114        px  144.77          $ PLUG 4               !!!
  115        px  266.000         $ cold detector plane  !!!
  116        px  266.000         $ cold detector plane  !!!
c *** Z CYLINDERS **************************************************************
   30        cz 22.85    $ CORE/REFLECTOR WALL
   31        cz 23.485   $ INNER REFLECTOR
   32        cz 33.02    $ RSR INNER WALL
   33        cz 33.665   $ RSR INNER WALL
   34        cz 37.465   $ RSR OUTER VOLUME
   35        cz 38.1     $ RSR OUTER WALL
   36        cz 53.35    $ OUTER REFLECTOR
   37        cz 53.975   $ OUTER REFLECTOR WALL
   38        cz 99.06    $ EDGE OF POOL               !!!
   39        cz 266.69   $ EDGE OF CONCRETE           !!!
c *** OTHER CYLINDERS **********************************************************
   40       c/x 0 -8.3 8.255   $ NE/SW BEAM PORT INNER
   90       c/x 0 -8.3 7.62    $ NE/SW BEAM PORT INNER
   41       c/x 0 -8.3 10.795  $ NE/SW BEAM PORT OUTER
   91       c/x 0 -8.3 10.16   $ NE/SW BEAM PORT OUTER
   43        cz 22             $ LOWER GRID PLATE RADIUS
   44        cz 24.75          $ UPPER GRID PLATE RADIUS
   45        cz 1.0            $ Bottom part of rabbit, above pin
   46       c/x 0 -8.3 1.27    $ Collimator Inner Diameter
   47       c/z 0 0 59         $ Outer detector surface
   48       c/z 0 0 24.485     $ Inner detector surface
c *** FLUX PROBE HOLES *********************************************************
   50        pz -12.4373   $ Pulse rod, span 39.37 cm
   51        pz -12.6811   $ Safety rod, span 33.02 cm
   52        pz -26.0390   $ Shim rod, span 43.18
   53        pz -21.3628   $ Regulating rod, span 41.91
   68        pz 24.257
c ** UNIVERSE FUEL PIN cylinders ***********************************************
  200        cz 0.2285   $ Zirc FILLer
  201        cz 0.635    $ Pin diameter
  202        cz 1.8161   $ Fuel diameter MARK III FUEL SCHEMATIC
  203        cz 1.985    $ CT & RABBIT
c ** UNIVERSE CONTROL ROD cylinders ********************************************
  210        cz 1.69165  $ Guide tube outer radius 1.89865 cm
  219        cz 1.58815  $ REG guide tube surface
  211        cz 1.511    $ Pulse meat
  212        cz 1.587    $ Pulse rod cladding surface
  213        cz 1.5113   $ Shim/safety meat
  214        cz 1.5875   $ Shim/safety rod cladding surface
  215        cz 1.03505  $ Rod reg meat
  216        cz 1.11125  $ Reg rod cladding surface
  217        cz 1.254    $ Extension tube outer radius
  218        cz 0.619    $ Extesion tube inner radius
c ** UNIVERSE SOURCE ***********************************************************
  230        cz 0.981  $ SOURCE HOLDER CAVITY
  231        cz 0.9525  $ SOURCE RADIUS
c source elevations ALSO USES SOME FUEL ELEVATIONS
  232        pz 3.8  $ Source cavity top
  233        pz -3.8  $ Source cavity bottom
  234        pz 2.55  $ Source heigth
c ******* END OF UNIVERSE SURFACE DEFINTIONS ***********************************
c ** A RING USES SURFACE 203 ***************************************************
c 1000      c/z 0 0 1.8669         $ A-1 = surface 203
c ** B Ring ********************************************************************
 1201       c/z 3.121883  2.586013 1.8669      $ B-1
 1202       c/z 3.800495  -1.410623 1.8669      $ B-2
 1203       c/z 0.678612  -3.996636 1.8669      $ B-3
 1204       c/z -3.121883  -2.586013 1.8669      $ B-4
 1205       c/z -3.800495  1.410623 1.8669      $ B-5
 1206       c/z -0.678612  3.996636 1.8669      $ B-6
 1301       c/z 6.145963  5.091011 1.8669      $ C-1
 1302       c/z 7.868066  1.335964 1.8669      $ C-2
 1303       c/z 7.481927  -2.777054 1.8669      $ C-3
 1304       c/z 5.091011  -6.145963 1.8669      $ C-4
 1305       c/z 1.335964  -7.868066 1.8669      $ C-5
 1306       c/z -2.777054  -7.481927 1.8669      $ C-6
 1307       c/z -6.145963  -5.091011 1.8669      $ C-7
 1308       c/z -7.868066  -1.335964 1.8669      $ C-8
 1309       c/z -7.481927  2.777054 1.8669      $ C-9
 1310       c/z -5.091011  6.145963 1.8669      $ C-10
 1311       c/z -1.335964  7.868066 1.8669      $ C-11
 1312       c/z 2.777054  7.481927 1.8669      $ C-12
 1401       c/z 9.199383  7.620314 1.8669      $ D-1
 1402       c/z 11.250894  4.014378 1.8669      $ D-2
 1403       c/z 11.945380  -0.075751 1.8669      $ D-3
 1404       c/z 11.199077  -4.156742 1.8669      $ D-4
 1405       c/z 9.102000  -7.736370 1.8669      $ D-5
 1406       c/z 5.907088  -10.382877 1.8669      $ D-6
 1407       c/z 1.999694  -11.777056 1.8669      $ D-7
 1408       c/z -2.148893  -11.750749 1.8669      $ D-8
 1409       c/z -6.038292  -10.307127 1.8669      $ D-9
 1410       c/z -9.199383  -7.620314 1.8669      $ D-10
 1411       c/z -11.250894  -4.014378 1.8669      $ D-11
 1412       c/z -11.945380  0.075751 1.8669      $ D-12
 1413       c/z -11.199077  4.156742 1.8669      $ D-13
 1414       c/z -9.102000  7.736370 1.8669      $ D-14
 1415       c/z -5.907088  10.382877 1.8669      $ D-15
 1416       c/z -1.999694  11.777056 1.8669      $ D-16
 1417       c/z 2.148893  11.750749 1.8669      $ D-17
 1418       c/z 6.038292  10.307127 1.8669      $ D-18
 1501       c/z 12.256716  10.152857 1.8669      $ E-1
 1502       c/z 14.466832  6.634635 1.8669      $ E-2
 1503       c/z 15.691056  2.664274 1.8669      $ E-3
 1504       c/z 15.845961  -1.487653 1.8669      $ E-4
 1505       c/z 14.920990  -5.538199 1.8669      $ E-5
 1506       c/z 12.979179  -9.211326 1.8669      $ E-6
 1507       c/z 10.152857  -12.256716 1.8669      $ E-7
 1508       c/z 6.634635  -14.466832 1.8669      $ E-8
 1509       c/z 2.664274  -15.691056 1.8669      $ E-9
 1510       c/z -1.487653  -15.845961 1.8669      $ E-10
 1511       c/z -5.538199  -14.920990 1.8669      $ E-11
 1512       c/z -9.211326  -12.979179 1.8669      $ E-12
 1513       c/z -12.256716  -10.152857 1.8669      $ E-13
 1514       c/z -14.466832  -6.634635 1.8669      $ E-14
 1515       c/z -15.691056  -2.664274 1.8669      $ E-15
 1516       c/z -15.845961  1.487653 1.8669      $ E-16
 1517       c/z -14.920990  5.538199 1.8669      $ E-17
 1518       c/z -12.979179  9.211326 1.8669      $ E-18
 1519       c/z -10.152857  12.256716 1.8669      $ E-19
 1520       c/z -6.634635  14.466832 1.8669      $ E-20
 1521       c/z -2.664274  15.691056 1.8669      $ E-21
 1522       c/z 1.487653  15.845961 1.8669      $ E-22
 1523       c/z 5.538199  14.920990 1.8669      $ E-23
 1524       c/z 9.211326  12.979179 1.8669      $ E-24
 1601       c/z 15.316005  12.687021 1.8669      $ F-1
 1602       c/z 17.619093  9.225402 1.8669      $ F-2
 1603       c/z 19.152143  5.360590 1.8669      $ F-3
 1604       c/z 19.848152  1.261493 1.8669      $ F-4
 1605       c/z 19.676701  -2.892736 1.8669      $ F-5
 1606       c/z 18.645284  -6.920539 1.8669      $ F-6
 1607       c/z 16.798979  -10.645881 1.8669      $ F-7
 1608       c/z 14.218478  -13.905947 1.8669      $ F-8
 1609       c/z 11.016561  -16.558257 1.8669      $ F-9
 1610       c/z 7.333167  -18.486891 1.8669      $ F-10
 1611       c/z 3.329279  -19.607559 1.8669      $ F-11
 1612       c/z -0.820114  -19.871283 1.8669      $ F-12
 1613       c/z -4.933665  -19.266538 1.8669      $ F-13
 1614       c/z -8.831591  -17.819751 1.8669      $ F-14
 1615       c/z -12.343534  -15.594155 1.8669      $ F-15
 1616       c/z -15.316005  -12.687021 1.8669      $ F-16
 1617       c/z -17.619093  -9.225402 1.8669      $ F-17
 1618       c/z -19.152143  -5.360590 1.8669      $ F-18
 1619       c/z -19.848152  -1.261493 1.8669      $ F-19
 1620       c/z -19.676701  2.892736 1.8669      $ F-20
 1621       c/z -18.645284  6.920539 1.8669      $ F-21
 1622       c/z -16.798979  10.645881 1.8669      $ F-22
 1623       c/z -14.218478  13.905947 1.8669      $ F-23
 1624       c/z -11.016561  16.558257 1.8669      $ F-24
 1625       c/z -7.333167  18.486891 1.8669      $ F-25
 1626       c/z -3.329279  19.607559 1.8669      $ F-26
 1627       c/z 0.820114  19.871283 1.8669      $ F-27
 1628       c/z 4.933665  19.266538 1.8669      $ F-28
 1629       c/z 8.831591  17.819751 1.8669      $ F-29
 1630       c/z 12.343534  15.594155 1.8669      $ F-30

tr1  0 0 0 0.6428 0.776 0 -0.776 0.6428 0 0 0 1
mode  n
c kcode 100000 1.000000 30 250
c    MATERIAL CARDS
c    U + Zr-H  rho=5.7473 g/cm^3 BOL
c    S(a,b)- Zr-H @ 300 K
c ******************************************************************************
c This current model accounts for the burnup of U235 and isotopes produced
c **  B-1  Fuel ID= 6315
m2011 62152.71c -1.18153981671e-06
      62151.71c -9.88307536455e-07
      62150.71c -2.7314773542e-06
      64155.71c -3.24283170638e-09
      63153.71c -5.16374263222e-07
      60143.71c -1.64557990772e-05
      60145.71c -1.188656831e-05
      95241.71c -2.59228393238e-10
      95243.71c -3.29318484526e-13
      94239.71c -4.43907769016e-05
      94238.71c -9.57105811593e-10
      63154.71c -6.09507436669e-09
      44103.71c -1.1409208552e-06
      92238.71c -0.0626485273293
      54135.71c -2.39220945459e-08
      62149.71c -6.29632194408e-07
      54131.71c -7.59510185155e-06
      1001.71c -0.0171082385555
      62147.71c -7.2168021021e-07
      92234.71c -1.30917969775e-08
      92235.71c -0.0150926373561
      92236.71c -9.26175836566e-05
      43099.71c -1.2453227246e-05
      42095.71c -7.58750779135e-06
      61147.71c -5.77852946396e-06
      40090.71c -0.904905118491
      45103.71c -5.40199915928e-06
      55133.71c -1.81032152433e-05
      93237.71c -1.64854642787e-07
      55135.71c -1.41454991415e-05
      94240.71c -9.18634751872e-07
      94241.71c -2.12911911447e-08
      94242.71c -1.79384409736e-10
m2012 62152.71c -1.56197635754e-06
      62151.71c -1.16498776203e-06
      62150.71c -3.58433342565e-06
      64155.71c -3.48699753693e-09
      63153.71c -6.56798523578e-07
      60143.71c -2.06157550046e-05
      60145.71c -1.49164436536e-05
      95241.71c -6.77435780786e-10
      95243.71c -1.60987341785e-12
      94239.71c -6.33903503677e-05
      94238.71c -2.10534941476e-09
      63154.71c -1.01217373786e-08
      44103.71c -1.43491510515e-06
      92238.71c -0.0626328913937
      54135.71c -2.8360277105e-08
      62149.71c -6.42194708036e-07
      54131.71c -9.53081626025e-06
      1001.71c -0.0171099437095
      62147.71c -9.04205580117e-07
      92234.71c -1.84831183046e-08
      92235.71c -0.0149446313575
      92236.71c -0.000117999682869
      43099.71c -1.56357705841e-05
      42095.71c -9.53780128474e-06
      61147.71c -7.22777224242e-06
      40090.71c -0.904995309108
      45103.71c -6.7889574932e-06
      55133.71c -2.27279803279e-05
      93237.71c -2.78011399044e-07
      55135.71c -1.68308696218e-05
      94240.71c -1.67541013129e-06
      94241.71c -5.55663586739e-08
      94242.71c -5.95946520263e-10
m2013 62152.71c -1.94974677102e-06
      62151.71c -1.30732440542e-06
      62150.71c -4.40963466639e-06
      64155.71c -3.60564454741e-09
      63153.71c -7.91398881127e-07
      60143.71c -2.45624758009e-05
      60145.71c -1.78022147036e-05
      95241.71c -1.15028100733e-09
      95243.71c -3.97038822185e-12
      94239.71c -7.51343658661e-05
      94238.71c -3.34285985641e-09
      63154.71c -1.43988650595e-08
      44103.71c -1.71333376525e-06
      92238.71c -0.0626245714538
      54135.71c -3.20112721282e-08
      62149.71c -6.41704101726e-07
      54131.71c -1.1372376497e-05
      1001.71c -0.0171115748573
      62147.71c -1.07810831543e-06
      92234.71c -2.17534684614e-08
      92235.71c -0.0148048082923
      92236.71c -0.000140791292457
      43099.71c -1.86681272514e-05
      42095.71c -1.13999284131e-05
      61147.71c -8.60543362776e-06
      40090.71c -0.905081585318
      45103.71c -8.10684940855e-06
      55133.71c -2.71336439067e-05
      93237.71c -3.7454981156e-07
      55135.71c -1.90679904411e-05
      94240.71c -2.37783008207e-06
      94241.71c -9.4266086722e-08
      94242.71c -1.21723604953e-09
m2014 62152.71c -2.0894668155e-06
      62151.71c -1.35218650119e-06
      62150.71c -4.69785311662e-06
      64155.71c -3.63131597618e-09
      63153.71c -8.38834130098e-07
      60143.71c -2.59330534964e-05
      60145.71c -1.88065294283e-05
      95241.71c -1.35080653364e-09
      95243.71c -5.19232204462e-12
      94239.71c -7.89872594894e-05
      94238.71c -3.85479333267e-09
      63154.71c -1.60507195513e-08
      44103.71c -1.80985360425e-06
      92238.71c -0.0626218740312
      54135.71c -3.3177656737e-08
      62149.71c -6.41090171931e-07
      54131.71c -1.20133569609e-05
      1001.71c -0.0171121437146
      62147.71c -1.13871626449e-06
      92234.71c -2.29336916552e-08
      92235.71c -0.0147561898443
      92236.71c -0.000148681413644
      43099.71c -1.97240197234e-05
      42095.71c -1.20505354165e-05
      61147.71c -9.08331927938e-06
      40090.71c -0.90511167386
      45103.71c -8.56613509971e-06
      55133.71c -2.86675889158e-05
      93237.71c -4.10247721152e-07
      55135.71c -1.97912109668e-05
      94240.71c -2.64278449739e-06
      94241.71c -1.10576723784e-07
      94242.71c -1.51341806413e-09
c **  B-2  Fuel ID= 10880
m2021 62152.71c -2.71618598977e-07
      62151.71c -3.4189136929e-07
      62150.71c -4.4293453591e-07
      64155.71c -4.52118115996e-10
      63153.71c -1.41418571818e-07
      60143.71c -3.90006308233e-06
      60145.71c -3.38555879369e-06
      95241.71c -1.5223827778e-12
      95243.71c -5.07094781491e-16
      94239.71c -1.28120537103e-05
      94238.71c -3.89669603659e-11
      63154.71c -4.75210550457e-10
      44103.71c -9.10161406123e-07
      92238.71c -0.062666117657
      54135.71c -2.40021971939e-08
      62149.71c -4.88525194837e-07
      54131.71c -1.96379641107e-06
      1001.71c -0.0171035996837
      62147.71c -4.84919307385e-08
      92234.71c -3.79993954435e-09
      92235.71c -0.0155037592909
      92236.71c -2.65182292267e-05
      43099.71c -3.43479946148e-06
      42095.71c -6.17689243897e-07
      61147.71c -1.58056030563e-06
      40090.71c -0.904659754902
      45103.71c -9.5687453818e-07
      55133.71c -4.80361957587e-06
      93237.71c -2.45061908578e-08
      55135.71c -4.02519720252e-06
      94240.71c -7.12537600578e-08
      94241.71c -4.52720184358e-10
      94242.71c -1.01326100009e-12
m2022 62152.71c -3.52311255755e-07
      62151.71c -4.2411720694e-07
      62150.71c -6.3707724123e-07
      64155.71c -5.4201944209e-10
      63153.71c -1.79426735499e-07
      60143.71c -4.93039209246e-06
      60145.71c -4.28148027032e-06
      95241.71c -4.09430639286e-12
      95243.71c -2.57103359525e-15
      94239.71c -1.85945186021e-05
      94238.71c -7.90338871022e-11
      63154.71c -8.0194364676e-10
      44103.71c -1.15236432499e-06
      92238.71c -0.0626616931881
      54135.71c -2.87166128828e-08
      62149.71c -5.42082206034e-07
      54131.71c -2.4839459701e-06
      1001.71c -0.0171041197293
      62147.71c -6.129384474e-08
      92234.71c -5.36456834028e-09
      92235.71c -0.0154597645595
      92236.71c -3.40822192127e-05
      43099.71c -4.34462022027e-06
      42095.71c -7.81608697733e-07
      61147.71c -1.99707230925e-06
      40090.71c -0.904687261645
      45103.71c -1.21108969277e-06
      55133.71c -6.07590798417e-06
      93237.71c -3.77914967406e-08
      55135.71c -4.8216760482e-06
      94240.71c -1.33153837802e-07
      94241.71c -1.2176376398e-09
      94242.71c -3.47554275897e-12
m2023 62152.71c -4.32410327808e-07
      62151.71c -4.98764621175e-07
      62150.71c -8.3977161524e-07
      64155.71c -6.16540119562e-10
      63153.71c -2.15514074021e-07
      60143.71c -5.90960793152e-06
      60145.71c -5.13324664743e-06
      95241.71c -7.10441382202e-12
      95243.71c -6.48110263661e-15
      94239.71c -2.23239775448e-05
      94238.71c -1.19265309518e-10
      63154.71c -1.15274875626e-09
      44103.71c -1.38211988771e-06
      92238.71c -0.0626594850599
      54135.71c -3.26340937379e-08
      62149.71c -5.74891184984e-07
      54131.71c -2.97818747811e-06
      1001.71c -0.0171046120727
      62147.71c -7.34679763859e-08
      92234.71c -6.39762310785e-09
      92235.71c -0.0154183185818
      92236.71c -4.08706997835e-05
      43099.71c -5.20971285006e-06
      42095.71c -9.37699684008e-07
      61147.71c -2.39278038896e-06
      40090.71c -0.904713303139
      45103.71c -1.45250526719e-06
      55133.71c -7.28558290213e-06
      93237.71c -4.82761181725e-08
      55135.71c -5.48678502254e-06
      94240.71c -1.92098883783e-07
      94241.71c -2.11185478047e-09
      94242.71c -7.2482675235e-12
m2024 62152.71c -4.60897587971e-07
      62151.71c -5.23963833446e-07
      62150.71c -9.12525008107e-07
      64155.71c -6.40338265046e-10
      63153.71c -2.28061327613e-07
      60143.71c -6.24883471445e-06
      60145.71c -5.42864392458e-06
      95241.71c -8.41919991608e-12
      95243.71c -8.60583582613e-15
      94239.71c -2.36274213283e-05
      94238.71c -1.35580370297e-10
      63154.71c -1.28883976365e-09
      44103.71c -1.46193405216e-06
      92238.71c -0.0626586587811
      54135.71c -3.39005153384e-08
      62149.71c -5.83853013e-07
      54131.71c -3.14953940428e-06
      1001.71c -0.0171047841061
      62147.71c -7.7679365944e-08
      92234.71c -6.75937009373e-09
      92235.71c -0.0154039428106
      92236.71c -4.32218453772e-05
      43099.71c -5.50975870766e-06
      42095.71c -9.91753203811e-07
      61147.71c -2.52982622797e-06
      40090.71c -0.904722402494
      45103.71c -1.53616352788e-06
      55133.71c -7.7050509062e-06
      93237.71c -5.21788382203e-08
      55135.71c -5.70163127247e-06
      94240.71c -2.15001474514e-07
      94241.71c -2.50327568876e-09
      94242.71c -9.08790516042e-12
c **  B-3  Fuel ID= 6577
m2031 62152.71c -1.16923476372e-06
      62151.71c -9.81101093258e-07
      62150.71c -2.70133782528e-06
      64155.71c -3.20497438721e-09
      63153.71c -5.11626142899e-07
      60143.71c -1.6298924017e-05
      60145.71c -1.17800968434e-05
      95241.71c -2.54077688455e-10
      95243.71c -3.22449999999e-13
      94239.71c -4.39999049905e-05
      94238.71c -9.41008485971e-10
      63154.71c -6.00548289314e-09
      44103.71c -1.14013313338e-06
      92238.71c -0.0626487527637
      54135.71c -2.39214332974e-08
      62149.71c -6.29306141517e-07
      54131.71c -7.52462160531e-06
      1001.71c -0.0171081806162
      62147.71c -7.10916354949e-07
      92234.71c -1.29752929667e-08
      92235.71c -0.0150977858181
      92236.71c -9.17899324893e-05
      43099.71c -1.23402344142e-05
      42095.71c -7.48517758659e-06
      61147.71c -5.72845746907e-06
      40090.71c -0.904902053911
      45103.71c -5.34421959832e-06
      55133.71c -1.79366205578e-05
      93237.71c -1.62763720119e-07
      55135.71c -1.40188066145e-05
      94240.71c -9.05091085462e-07
      94241.71c -2.09061816554e-08
      94242.71c -1.75809751944e-10
m2032 62152.71c -1.54555160648e-06
      62151.71c -1.1569795796e-06
      62150.71c -3.54621370959e-06
      64155.71c -3.44852719546e-09
      63153.71c -6.50726471929e-07
      60143.71c -2.04201661485e-05
      60145.71c -1.47833844474e-05
      95241.71c -6.63993656492e-10
      95243.71c -1.57631869536e-12
      94239.71c -6.28394073895e-05
      94238.71c -2.06950298121e-09
      63154.71c -9.97339944257e-09
      44103.71c -1.43399854629e-06
      92238.71c -0.0626332598548
      54135.71c -2.83621554743e-08
      62149.71c -6.42123996053e-07
      54131.71c -9.44273397687e-06
      1001.71c -0.017109870992
      62147.71c -8.90741384719e-07
      92234.71c -1.83193841213e-08
      92235.71c -0.0149510744757
      92236.71c -0.000116950392609
      43099.71c -1.54944335903e-05
      42095.71c -9.40934277894e-06
      61147.71c -7.16555964659e-06
      40090.71c -0.904991462868
      45103.71c -6.71653325005e-06
      55133.71c -2.25196019489e-05
      93237.71c -2.74384950921e-07
      55135.71c -1.66806846618e-05
      94240.71c -1.65086173009e-06
      94241.71c -5.45645962579e-08
      94242.71c -5.84086545085e-10
m2033 62152.71c -1.92910666795e-06
      62151.71c -1.2988689384e-06
      62150.71c -4.3640254637e-06
      64155.71c -3.56798430999e-09
      63153.71c -7.84052959548e-07
      60143.71c -2.43304006862e-05
      60145.71c -1.76439296374e-05
      95241.71c -1.12748238328e-09
      95243.71c -3.88765308109e-12
      94239.71c -7.44894016495e-05
      94238.71c -3.2854866407e-09
      63154.71c -1.4188349141e-08
      44103.71c -1.71232089961e-06
      92238.71c -0.062625016765
      54135.71c -3.20160401136e-08
      62149.71c -6.41781621516e-07
      54131.71c -1.12676183121e-05
      1001.71c -0.017111488021
      62147.71c -1.06207503642e-06
      92234.71c -2.15620385692e-08
      92235.71c -0.014812470389
      92236.71c -0.000139543965673
      43099.71c -1.84998787718e-05
      42095.71c -1.12465632632e-05
      61147.71c -8.53174116085e-06
      40090.71c -0.905076992292
      45103.71c -8.02056351218e-06
      55133.71c -2.68856038723e-05
      93237.71c -3.69578434099e-07
      55135.71c -1.8898394483e-05
      94240.71c -2.34314665515e-06
      94241.71c -9.25709898646e-08
      94242.71c -1.19303529462e-09
m2034 62152.71c -2.06732046937e-06
      62151.71c -1.34363444725e-06
      62150.71c -4.64967791177e-06
      64155.71c -3.59409554533e-09
      63153.71c -8.31039924387e-07
      60143.71c -2.56884701888e-05
      60145.71c -1.86395753022e-05
      95241.71c -1.32405188298e-09
      95243.71c -5.08420281462e-12
      94239.71c -7.8312422211e-05
      94238.71c -3.7885266986e-09
      63154.71c -1.5816322173e-08
      44103.71c -1.80883316155e-06
      92238.71c -0.0626223428618
      54135.71c -3.31837977982e-08
      62149.71c -6.41201170226e-07
      54131.71c -1.19028601002e-05
      1001.71c -0.0171120520376
      62147.71c -1.12179034566e-06
      92234.71c -2.27323945391e-08
      92235.71c -0.0147642702948
      92236.71c -0.000147366464151
      43099.71c -1.95465102055e-05
      42095.71c -1.18885007052e-05
      61147.71c -9.00570926222e-06
      40090.71c -0.90510682479
      45103.71c -8.47505664845e-06
      55133.71c -2.84058996403e-05
      93237.71c -4.04786107465e-07
      55135.71c -1.96154339628e-05
      94240.71c -2.60431053474e-06
      94241.71c -1.08591677995e-07
      94242.71c -1.48334911282e-09
c **  B-4  Fuel ID= 2966
m2041 62152.71c -1.15066085322e-06
      62151.71c -9.31517890363e-07
      62150.71c -2.61803974965e-06
      64155.71c -3.12001303908e-09
      63153.71c -4.96628514395e-07
      60143.71c -1.57354155642e-05
      60145.71c -1.14051186939e-05
      95241.71c -3.27135250919e-10
      95243.71c -4.58466276961e-13
      94239.71c -4.25939214174e-05
      94238.71c -1.04625994832e-09
      63154.71c -6.25749656644e-09
      44103.71c -1.10318835337e-06
      92238.71c -0.0626492590771
      54135.71c -2.38983384319e-08
      62149.71c -6.06340549197e-07
      54131.71c -7.27601549513e-06
      1001.71c -0.0171079743583
      62147.71c -7.41126198335e-07
      92234.71c -1.25570264025e-08
      92235.71c -0.0151159162069
      92236.71c -8.88999917731e-05
      43099.71c -1.19442026164e-05
      42095.71c -7.43749740974e-06
      61147.71c -5.4796626842e-06
      40090.71c -0.904891144326
      45103.71c -5.17502239564e-06
      55133.71c -1.73524963372e-05
      93237.71c -1.65768131194e-07
      55135.71c -1.35737355086e-05
      94240.71c -9.47915171699e-07
      94241.71c -2.43332420504e-08
      94242.71c -2.26347919669e-10
m2042 62152.71c -1.52162916592e-06
      62151.71c -1.09471123608e-06
      62150.71c -3.42833915273e-06
      64155.71c -3.32048744033e-09
      63153.71c -6.3240898812e-07
      60143.71c -1.97054166739e-05
      60145.71c -1.4309709485e-05
      95241.71c -8.49915022275e-10
      95243.71c -2.22395730477e-12
      94239.71c -6.08339073777e-05
      94238.71c -2.32262596247e-09
      63154.71c -1.03943685165e-08
      44103.71c -1.38813397647e-06
      92238.71c -0.062634111603
      54135.71c -2.8351358926e-08
      62149.71c -6.26006449985e-07
      54131.71c -9.12809386979e-06
      1001.71c -0.0171096098859
      62147.71c -9.27866633095e-07
      92234.71c -1.77345294104e-08
      92235.71c -0.0149740186043
      92236.71c -0.000113255523024
      43099.71c -1.49942026189e-05
      42095.71c -9.34506559897e-06
      61147.71c -6.8509380211e-06
      40090.71c -0.904977652194
      45103.71c -6.50195543995e-06
      55133.71c -2.17816618586e-05
      93237.71c -2.81380783906e-07
      55135.71c -1.61479390848e-05
      94240.71c -1.72594671705e-06
      94241.71c -6.3152536654e-08
      94242.71c -7.48367986186e-10
m2043 62152.71c -1.90046042971e-06
      62151.71c -1.22572143442e-06
      62150.71c -4.21684052664e-06
      64155.71c -3.40732743456e-09
      63153.71c -7.63154062489e-07
      60143.71c -2.34832743534e-05
      60145.71c -1.70854407691e-05
      95241.71c -1.44084832951e-09
      95243.71c -5.51700393729e-12
      94239.71c -7.19778132364e-05
      94238.71c -3.72265671582e-09
      63154.71c -1.47977599178e-08
      44103.71c -1.65869402186e-06
      92238.71c -0.0626260849365
      54135.71c -3.20257455816e-08
      62149.71c -6.3131542893e-07
      54131.71c -1.0896074825e-05
      1001.71c -0.0171111805594
      62147.71c -1.10651631513e-06
      92234.71c -2.09901856675e-08
      92235.71c -0.0148395612651
      92236.71c -0.000135164654212
      43099.71c -1.79101636374e-05
      42095.71c -1.11735352658e-05
      61147.71c -8.15845368555e-06
      40090.71c -0.905060729743
      45103.71c -7.76678417733e-06
      55133.71c -2.60153741533e-05
      93237.71c -3.81479897878e-07
      55135.71c -1.82982259837e-05
      94240.71c -2.44464305017e-06
      94241.71c -1.0695889911e-07
      94242.71c -1.52786714606e-09
m2044 62152.71c -2.03635827759e-06
      62151.71c -1.26669558528e-06
      62150.71c -4.49180454972e-06
      64155.71c -3.42199255443e-09
      63153.71c -8.09163791288e-07
      60143.71c -2.47895637231e-05
      60145.71c -1.80480405562e-05
      95241.71c -1.69420517366e-09
      95243.71c -7.24106156759e-12
      94239.71c -7.56205755745e-05
      94238.71c -4.30905903224e-09
      63154.71c -1.64953216666e-08
      44103.71c -1.75271535792e-06
      92238.71c -0.0626235237775
      54135.71c -3.32058254487e-08
      62149.71c -6.32282363105e-07
      54131.71c -1.15090276848e-05
      1001.71c -0.017111725514
      62147.71c -1.16817292148e-06
      92234.71c -2.20876347153e-08
      92235.71c -0.01479296028
      92236.71c -0.000142744612246
      43099.71c -1.89218926076e-05
      42095.71c -1.18079216852e-05
      61147.71c -8.61057846915e-06
      40090.71c -0.905089553999
      45103.71c -8.20540423687e-06
      55133.71c -2.74841930678e-05
      93237.71c -4.19454631628e-07
      55135.71c -1.8991390336e-05
      94240.71c -2.71770352613e-06
      94241.71c -1.25752089214e-07
      94242.71c -1.90497867175e-09
c **  B-5  Fuel ID= 10895
m2051 62152.71c -3.23740171924e-07
      62151.71c -2.60088770363e-07
      62150.71c -7.56948384356e-07
      64155.71c -9.2237580645e-10
      63153.71c -1.39335676537e-07
      60143.71c -4.46888985567e-06
      60145.71c -3.19669233094e-06
      95241.71c -7.89177864279e-11
      95243.71c -9.78955882358e-14
      94239.71c -1.19383486632e-05
      94238.71c -2.82991808107e-10
      63154.71c -1.76053898334e-09
      44103.71c -2.64756362539e-07
      92238.71c -0.0626661592644
      54135.71c -5.50217834681e-09
      62149.71c -1.48501845533e-07
      54131.71c -2.05360942481e-06
      1001.71c -0.0171034279528
      62147.71c -2.11705210698e-07
      92234.71c -3.54464176705e-09
      92235.71c -0.0155134469233
      92236.71c -2.49127905983e-05
      43099.71c -3.35545384568e-06
      42095.71c -2.19009296722e-06
      61147.71c -1.54839921377e-06
      40090.71c -0.904650671552
      45103.71c -1.49467127199e-06
      55133.71c -4.88833205878e-06
      93237.71c -4.67556940933e-08
      55135.71c -3.81096844251e-06
      94240.71c -2.65725286777e-07
      94241.71c -6.35621585389e-09
      94242.71c -5.36335991555e-11
m2052 62152.71c -4.32549196272e-07
      62151.71c -3.06242382858e-07
      62150.71c -9.96323269993e-07
      64155.71c -9.85470742026e-10
      63153.71c -1.78687838383e-07
      60143.71c -5.63544564066e-06
      60145.71c -4.03882347472e-06
      95241.71c -2.09906251664e-10
      95243.71c -4.94232648285e-13
      94239.71c -1.7149908597e-05
      94238.71c -6.30892897243e-10
      63154.71c -2.96315908133e-09
      44103.71c -3.35385737418e-07
      92238.71c -0.0626618327951
      54135.71c -6.55457716339e-09
      62149.71c -1.50176546126e-07
      54131.71c -2.59443178417e-06
      1001.71c -0.0171039011553
      62147.71c -2.66993681762e-07
      92234.71c -4.95249048186e-09
      92235.71c -0.0154723411721
      92236.71c -3.19445745434e-05
      43099.71c -4.24184844405e-06
      42095.71c -2.77175531516e-06
      61147.71c -1.94921092704e-06
      40090.71c -0.904675700625
      45103.71c -1.8912652037e-06
      55133.71c -6.17905565184e-06
      93237.71c -7.97888493763e-08
      55135.71c -4.55722385397e-06
      94240.71c -4.91190734605e-07
      94241.71c -1.68908176977e-08
      94242.71c -1.8285362077e-10
m2053 62152.71c -5.41977295886e-07
      62151.71c -3.41930818098e-07
      62150.71c -1.22327177667e-06
      64155.71c -1.01102784945e-09
      63153.71c -2.15872411573e-07
      60143.71c -6.72440964408e-06
      60145.71c -4.82822690151e-06
      95241.71c -3.57237564193e-10
      95243.71c -1.22509348786e-12
      94239.71c -2.0298234754e-05
      94238.71c -1.0106452901e-09
      63154.71c -4.23267307955e-09
      44103.71c -4.00980493267e-07
      92238.71c -0.0626596227823
      54135.71c -7.40117535085e-09
      62149.71c -1.49404364098e-07
      54131.71c -3.10077833613e-06
      1001.71c -0.0171043457733
      62147.71c -3.18916047618e-07
      92234.71c -5.87953642877e-09
      92235.71c -0.0154341084124
      92236.71c -3.81706788886e-05
      43099.71c -5.07306488534e-06
      42095.71c -3.31911042215e-06
      61147.71c -2.32398975481e-06
      40090.71c -0.90469921778
      45103.71c -2.26227382314e-06
      55133.71c -7.38933422812e-06
      93237.71c -1.08324412441e-07
      55135.71c -5.16808534096e-06
      94240.71c -6.97391120855e-07
      94241.71c -2.87274745974e-08
      94242.71c -3.75148628605e-10
m2054 62152.71c -5.81330754471e-07
      62151.71c -3.53056686796e-07
      62150.71c -1.30239586034e-06
      64155.71c -1.0157711964e-09
      63153.71c -2.29005695476e-07
      60143.71c -7.10203639371e-06
      60145.71c -5.10266667758e-06
      95241.71c -4.20201390918e-10
      95243.71c -1.61306076444e-12
      94239.71c -2.13342375009e-05
      94238.71c -1.17068828207e-09
      63154.71c -4.72384525142e-09
      44103.71c -4.23768440799e-07
      92238.71c -0.0626588453751
      54135.71c -7.67170774101e-09
      62149.71c -1.49070986042e-07
      54131.71c -3.27673064313e-06
      1001.71c -0.0171045017493
      62147.71c -3.36959053413e-07
      92234.71c -6.22360937023e-09
      92235.71c -0.0154208191785
      92236.71c -4.03313468342e-05
      43099.71c -5.3621977941e-06
      42095.71c -3.50987142023e-06
      61147.71c -2.45394449682e-06
      40090.71c -0.904707467811
      45103.71c -2.391344411e-06
      55133.71c -7.81030894376e-06
      93237.71c -1.19107767036e-07
      55135.71c -5.36518551718e-06
      94240.71c -7.75842458185e-07
      94241.71c -3.37826586177e-08
      94242.71c -4.67793633568e-10
c **  B-6  Fuel ID= 11341
m2061 62152.71c -1.16271708654e-07
      62151.71c -1.59149270935e-07
      62150.71c -1.05853027243e-07
      64155.71c -1.05870283813e-10
      63153.71c -6.18050922012e-08
      60143.71c -1.27385032192e-06
      60145.71c -1.5341558545e-06
      95241.71c -6.02792542289e-14
      95243.71c -9.78840086264e-18
      94239.71c -5.585959338e-06
      94238.71c -5.0753657949e-12
      63154.71c -9.14553399019e-11
      44103.71c -5.95529523966e-07
      92238.71c -0.0626698330477
      54135.71c -2.39564894401e-08
      62149.71c -2.98889454169e-07
      54131.71c -7.45625290222e-07
      1001.71c -0.0171025743566
      62147.71c -7.2896451294e-09
      92234.71c -1.71627855768e-09
      92235.71c -0.0155933499341
      92236.71c -1.20727410015e-05
      43099.71c -1.47353649396e-06
      42095.71c -8.30381026031e-08
      61147.71c -5.77022282035e-07
      40090.71c -0.904605522335
      45103.71c -2.5382275059e-07
      55133.71c -1.91362566045e-06
      93237.71c -8.00085215384e-09
      55135.71c -1.81459122207e-06
      94240.71c -1.36559460625e-08
      94241.71c -3.84441142234e-11
      94242.71c -3.97738957606e-14
m2062 62152.71c -1.49211076521e-07
      62151.71c -1.9990018793e-07
      62150.71c -1.59655437157e-07
      64155.71c -1.30857664031e-10
      63153.71c -7.84291719646e-08
      60143.71c -1.61366319302e-06
      60145.71c -1.9436957194e-06
      95241.71c -1.63561390286e-13
      95243.71c -5.01773163208e-17
      94239.71c -8.15081407314e-06
      94238.71c -1.00685106684e-11
      63154.71c -1.54940039591e-10
      44103.71c -7.55078473357e-07
      92238.71c -0.0626678355538
      54135.71c -2.87221099308e-08
      62149.71c -3.53496655322e-07
      54131.71c -9.44859497092e-07
      1001.71c -0.0171028213855
      62147.71c -9.2333636231e-09
      92234.71c -2.43715269225e-09
      92235.71c -0.0155731537684
      92236.71c -1.55463670468e-05
      43099.71c -1.86711541669e-06
      42095.71c -1.05230129799e-07
      61147.71c -7.3078177304e-07
      40090.71c -0.904618588418
      45103.71c -3.21770064937e-07
      55133.71c -2.42474388999e-06
      93237.71c -1.19196260706e-08
      55135.71c -2.17760493545e-06
      94240.71c -2.5744734049e-08
      94241.71c -1.04325752586e-10
      94242.71c -1.37547327913e-13
m2063 62152.71c -1.81449476496e-07
      62151.71c -2.3805094496e-07
      62150.71c -2.19908859344e-07
      64155.71c -1.53265713336e-10
      63153.71c -9.42940656795e-08
      60143.71c -1.93865678376e-06
      60145.71c -2.33535266223e-06
      95241.71c -2.85047351608e-13
      95243.71c -1.26985220462e-16
      94239.71c -9.81628544322e-06
      94238.71c -1.48735956762e-11
      63154.71c -2.23769020552e-10
      44103.71c -9.07419949947e-07
      92238.71c -0.0626670099748
      54135.71c -3.27167232819e-08
      62149.71c -3.96920621878e-07
      54131.71c -1.13528927481e-06
      1001.71c -0.0171030530499
      62147.71c -1.10919335126e-08
      92234.71c -2.92073889396e-09
      92235.71c -0.0155540165667
      92236.71c -1.86848181795e-05
      43099.71c -2.24353469428e-06
      42095.71c -1.26468559703e-07
      61147.71c -8.77759289016e-07
      40090.71c -0.904630841831
      45103.71c -3.86662859721e-07
      55133.71c -2.91353615909e-06
      93237.71c -1.47813744675e-08
      55135.71c -2.48280435302e-06
      94240.71c -3.72804192391e-08
      94241.71c -1.81845973293e-10
      94242.71c -2.8818836498e-13
m2064 62152.71c -1.930615462e-07
      62151.71c -2.5142527661e-07
      62150.71c -2.42884725506e-07
      64155.71c -1.60936083495e-10
      63153.71c -9.99289423834e-08
      60143.71c -2.05401648572e-06
      60145.71c -2.47436571428e-06
      95241.71c -3.38652988649e-13
      95243.71c -1.68162363369e-16
      94239.71c -1.03895448965e-05
      94238.71c -1.68485576983e-11
      63154.71c -2.50913919831e-10
      44103.71c -9.61499876822e-07
      92238.71c -0.062666671809
      54135.71c -3.40340909169e-08
      62149.71c -4.10760387045e-07
      54131.71c -1.20288167661e-06
      1001.71c -0.0171031368065
      62147.71c -1.17515045387e-08
      92234.71c -3.09489296304e-09
      92235.71c -0.0155472175426
      92236.71c -1.97929161819e-05
      43099.71c -2.37714939382e-06
      42095.71c -1.34010588697e-07
      61147.71c -9.29905289026e-07
      40090.71c -0.904635271967
      45103.71c -4.09700256488e-07
      55133.71c -3.08703550228e-06
      93237.71c -1.58615459198e-08
      55135.71c -2.58360313246e-06
      94240.71c -4.17976612981e-08
      94241.71c -2.16106552682e-10
      94242.71c -3.62581551739e-13
c **  C-1  Fuel ID= 11351
m3011 62152.71c -3.1161788883e-07
      62151.71c -2.5034988104e-07
      62150.71c -7.2860484408e-07
      64155.71c -8.87837922017e-10
      63153.71c -1.34118324282e-07
      60143.71c -4.30155458918e-06
      60145.71c -3.07699384198e-06
      95241.71c -7.59627507816e-11
      95243.71c -9.42299386282e-14
      94239.71c -1.14913233796e-05
      94238.71c -2.72395326396e-10
      63154.71c -1.69461651278e-09
      44103.71c -2.54842697644e-07
      92238.71c -0.0626664050138
      54135.71c -5.29615212785e-09
      62149.71c -1.42941270827e-07
      54131.71c -1.97671308333e-06
      1001.71c -0.0171033619187
      62147.71c -2.03778018712e-07
      92234.71c -3.41191449163e-09
      92235.71c -0.0155192423059
      92236.71c -2.39799440551e-05
      43099.71c -3.22981061401e-06
      42095.71c -2.10808607017e-06
      61147.71c -1.49042020702e-06
      40090.71c -0.90464717881
      45103.71c -1.4387040802e-06
      55133.71c -4.70529099619e-06
      93237.71c -4.50049513398e-08
      55135.71c -3.66826870264e-06
      94240.71c -2.55775341016e-07
      94241.71c -6.11821064273e-09
      94242.71c -5.16253168085e-11
m3012 62152.71c -4.16352615608e-07
      62151.71c -2.94775295415e-07
      62150.71c -9.59016461081e-07
      64155.71c -9.48570300403e-10
      63153.71c -1.71996964806e-07
      60143.71c -5.42442929688e-06
      60145.71c -3.88759182116e-06
      95241.71c -2.02046420768e-10
      95243.71c -4.75726362702e-13
      94239.71c -1.65077391504e-05
      94238.71c -6.07269439405e-10
      63154.71c -2.85220512396e-09
      44103.71c -3.22827392155e-07
      92238.71c -0.0626622405469
      54135.71c -6.3091444157e-09
      62149.71c -1.44553263124e-07
      54131.71c -2.49728462951e-06
      1001.71c -0.0171038174023
      62147.71c -2.56996241608e-07
      92234.71c -4.76704704034e-09
      92235.71c -0.0154796757382
      92236.71c -3.07484264917e-05
      43099.71c -4.08301462565e-06
      42095.71c -2.6679683727e-06
      61147.71c -1.87622373325e-06
      40090.71c -0.904671270683
      45103.71c -1.82044775751e-06
      55133.71c -5.9476841127e-06
      93237.71c -7.6801197229e-08
      55135.71c -4.38658096665e-06
      94240.71c -4.72798351905e-07
      94241.71c -1.62583497757e-08
      94242.71c -1.76006761629e-10
m3013 62152.71c -5.21683236697e-07
      62151.71c -3.29127395678e-07
      62150.71c -1.17746699845e-06
      64155.71c -9.73170435172e-10
      63153.71c -2.07789180908e-07
      60143.71c -6.47261760711e-06
      60145.71c -4.64743644542e-06
      95241.71c -3.43860988593e-10
      95243.71c -1.17922049661e-12
      94239.71c -1.95381778649e-05
      94238.71c -9.72802200559e-10
      63154.71c -4.07418282792e-09
      44103.71c -3.85965986339e-07
      92238.71c -0.0626601132869
      54135.71c -7.12404217244e-09
      62149.71c -1.43809995052e-07
      54131.71c -2.98467129703e-06
      1001.71c -0.0171042453718
      62147.71c -3.06974401362e-07
      92234.71c -5.65938023184e-09
      92235.71c -0.0154428745845
      92236.71c -3.67413975838e-05
      43099.71c -4.88310659772e-06
      42095.71c -3.19482805115e-06
      61147.71c -2.23696916189e-06
      40090.71c -0.90469390725
      45103.71c -2.17756415132e-06
      55133.71c -7.11264443442e-06
      93237.71c -1.04268260911e-07
      55135.71c -4.97456906159e-06
      94240.71c -6.71277671469e-07
      94241.71c -2.76517891872e-08
      94242.71c -3.61101382473e-10
m3014 62152.71c -5.59563125405e-07
      62151.71c -3.39836661985e-07
      62150.71c -1.25362832177e-06
      64155.71c -9.77736169953e-10
      63153.71c -2.20430695796e-07
      60143.71c -6.83610431864e-06
      60145.71c -4.91159996618e-06
      95241.71c -4.04467167431e-10
      95243.71c -1.55266053941e-12
      94239.71c -2.05353880254e-05
      94238.71c -1.12685246557e-09
      63154.71c -4.54696331217e-09
      44103.71c -4.07900650977e-07
      92238.71c -0.0626593649893
      54135.71c -7.38444461734e-09
      62149.71c -1.43489100165e-07
      54131.71c -3.15403516101e-06
      1001.71c -0.0171043955074
      62147.71c -3.24341796147e-07
      92234.71c -5.99056954018e-09
      92235.71c -0.0154300829591
      92236.71c -3.8821160437e-05
      43099.71c -5.16141307444e-06
      42095.71c -3.37844610244e-06
      61147.71c -2.36205781588e-06
      40090.71c -0.904701848362
      45103.71c -2.30180175785e-06
      55133.71c -7.51785596983e-06
      93237.71c -1.14647838378e-07
      55135.71c -5.16428892378e-06
      94240.71c -7.46791439098e-07
      94241.71c -3.25176844597e-08
      94242.71c -4.502773432e-10
c **  C-2  Fuel ID= 6316
m3021 62152.71c -1.13420436881e-06
      62151.71c -9.60585574033e-07
      62150.71c -2.61553570789e-06
      64155.71c -3.09720103246e-09
      63153.71c -4.98109051161e-07
      60143.71c -1.58523273776e-05
      60145.71c -1.14769906701e-05
      95241.71c -2.3941450717e-10
      95243.71c -3.02896591676e-13
      94239.71c -4.28871590787e-05
      94238.71c -8.95182135265e-10
      63154.71c -5.75043119178e-09
      44103.71c -1.13789062319e-06
      92238.71c -0.0626493945373
      54135.71c -2.39195508353e-08
      62149.71c -6.28377924341e-07
      54131.71c -7.3239763206e-06
      1001.71c -0.0171080156729
      62147.71c -6.80273488068e-07
      92234.71c -1.26436258448e-08
      92235.71c -0.0151124426145
      92236.71c -8.94337502545e-05
      43099.71c -1.20185630223e-05
      42095.71c -7.19386088208e-06
      61147.71c -5.58591100809e-06
      40090.71c -0.904893329573
      45103.71c -5.17973100648e-06
      55133.71c -1.7462353797e-05
      93237.71c -1.56811218584e-07
      55135.71c -1.36581345183e-05
      94240.71c -8.66534568601e-07
      94241.71c -1.98101250621e-08
      94242.71c -1.65633308636e-10
m3022 62152.71c -1.49879313119e-06
      62151.71c -1.13418164502e-06
      62150.71c -3.43769335545e-06
      64155.71c -3.33900867035e-09
      63153.71c -6.33440372675e-07
      60143.71c -1.98633579107e-05
      60145.71c -1.44045874973e-05
      95241.71c -6.25726212779e-10
      95243.71c -1.48079410222e-12
      94239.71c -6.12709663498e-05
      94238.71c -1.96745427607e-09
      63154.71c -9.55110654502e-09
      44103.71c -1.43138925896e-06
      92238.71c -0.0626343088009
      54135.71c -2.83675028724e-08
      62149.71c -6.41922691053e-07
      54131.71c -9.19197868345e-06
      1001.71c -0.0171096639778
      62147.71c -8.52411108326e-07
      92234.71c -1.78532607238e-08
      92235.71c -0.0149694169384
      92236.71c -0.000113963241539
      43099.71c -1.50920711857e-05
      42095.71c -9.04364324123e-06
      61147.71c -6.98845095774e-06
      40090.71c -0.904980513276
      45103.71c -6.51035373646e-06
      55133.71c -2.19263841124e-05
      93237.71c -2.6406106934e-07
      55135.71c -1.62531336007e-05
      94240.71c -1.58097660327e-06
      94241.71c -5.17127488838e-08
      94242.71c -5.50323211061e-10
m3023 62152.71c -1.87034780178e-06
      62151.71c -1.27479766057e-06
      62150.71c -4.23418381377e-06
      64155.71c -3.46077201349e-09
      63153.71c -7.63140369071e-07
      60143.71c -2.36697222701e-05
      60145.71c -1.71933189505e-05
      95241.71c -1.06257857469e-09
      95243.71c -3.65212019411e-12
      94239.71c -7.26532981203e-05
      94238.71c -3.12215484473e-09
      63154.71c -1.35890460929e-08
      44103.71c -1.70943744326e-06
      92238.71c -0.0626262844905
      54135.71c -3.20296137579e-08
      62149.71c -6.42002307186e-07
      54131.71c -1.09693895608e-05
      1001.71c -0.0171112408129
      62147.71c -1.0164310155e-06
      92234.71c -2.10170701947e-08
      92235.71c -0.0148342830762
      92236.71c -0.000135993038268
      43099.71c -1.80209039399e-05
      42095.71c -1.08099587419e-05
      61147.71c -8.32195122972e-06
      40090.71c -0.905063916728
      45103.71c -7.77492222734e-06
      55133.71c -2.61794760424e-05
      93237.71c -3.55425767229e-07
      55135.71c -1.84155836087e-05
      94240.71c -2.24440883229e-06
      94241.71c -8.77453371507e-08
      94242.71c -1.12413985768e-09
m3024 62152.71c -2.00427358513e-06
      62151.71c -1.3192882029e-06
      62150.71c -4.51253128972e-06
      64155.71c -3.48813530337e-09
      63153.71c -8.08851145155e-07
      60143.71c -2.49921830729e-05
      60145.71c -1.81642852751e-05
      95241.71c -1.24788593882e-09
      95243.71c -4.77640573896e-12
      94239.71c -7.63912751509e-05
      94238.71c -3.59987685177e-09
      63154.71c -1.51490326669e-08
      44103.71c -1.80592813459e-06
      92238.71c -0.0626236775427
      54135.71c -3.32012803557e-08
      62149.71c -6.41517163509e-07
      54131.71c -1.1588294314e-05
      1001.71c -0.0171117910485
      62147.71c -1.0736051309e-06
      92234.71c -2.21593358548e-08
      92235.71c -0.0147872739644
      92236.71c -0.000143623026386
      43099.71c -1.90411707714e-05
      42095.71c -1.14272154159e-05
      61147.71c -8.78476673124e-06
      40090.71c -0.905093020311
      45103.71c -8.21577177421e-06
      55133.71c -2.76609147406e-05
      93237.71c -3.8923782136e-07
      55135.71c -1.91150267014e-05
      94240.71c -2.49478170075e-06
      94241.71c -1.02940589943e-07
      94242.71c -1.39774791793e-09
c **  C-3  Fuel ID= 2963
m3031 62152.71c -1.05430400247e-06
      62151.71c -8.71263092421e-07
      62150.71c -2.38053395484e-06
      64155.71c -2.82741058509e-09
      63153.71c -4.58650072417e-07
      60143.71c -1.44742913481e-05
      60145.71c -1.05501669963e-05
      95241.71c -2.87790635549e-10
      95243.71c -4.01686145899e-13
      94239.71c -3.95057726474e-05
      94238.71c -9.3133237782e-10
      63154.71c -5.59149502134e-09
      44103.71c -1.09076372498e-06
      92238.71c -0.0626510344414
      54135.71c -2.39032710193e-08
      62149.71c -5.99312099456e-07
      54131.71c -6.70986519932e-06
      1001.71c -0.0171075077774
      62147.71c -6.61178921618e-07
      92234.71c -1.16912526897e-08
      92235.71c -0.0151572393497
      92236.71c -8.22698554988e-05
      43099.71c -1.10368643366e-05
      42095.71c -6.6643548239e-06
      61147.71c -5.07048988262e-06
      40090.71c -0.904866465489
      45103.71c -4.71720411017e-06
      55133.71c -1.60147306218e-05
      93237.71c -1.50461873125e-07
      55135.71c -1.25584456767e-05
      94240.71c -8.47515221781e-07
      94241.71c -2.14860963457e-08
      94242.71c -1.98956499034e-10
m3032 62152.71c -1.39523585911e-06
      62151.71c -1.02758155223e-06
      62150.71c -3.12980159019e-06
      64155.71c -3.01937921661e-09
      63153.71c -5.84585498403e-07
      60143.71c -1.81521602542e-05
      60145.71c -1.32556857447e-05
      95241.71c -7.50786696645e-10
      95243.71c -1.97434308066e-12
      94239.71c -5.64043069972e-05
      94238.71c -2.06620967948e-09
      63154.71c -9.30494872896e-09
      44103.71c -1.37455185834e-06
      92238.71c -0.0626369868943
      54135.71c -2.83992010127e-08
      62149.71c -6.22162412505e-07
      54131.71c -8.42979481033e-06
      1001.71c -0.0171090344672
      62147.71c -8.28792804213e-07
      92234.71c -1.64182319091e-08
      92235.71c -0.0150250653646
      92236.71c -0.000104946420297
      43099.71c -1.38745063335e-05
      42095.71c -8.38414311515e-06
      61147.71c -6.34895214145e-06
      40090.71c -0.904947216609
      45103.71c -5.9344685506e-06
      55133.71c -2.01302829122e-05
      93237.71c -2.54785205941e-07
      55135.71c -1.49566214832e-05
      94240.71c -1.54522294265e-06
      94241.71c -5.5980224734e-08
      94242.71c -6.61374574747e-10
m3033 62152.71c -1.74154298769e-06
      62151.71c -1.15365813485e-06
      62150.71c -3.8570738555e-06
      64155.71c -3.10717066904e-09
      63153.71c -7.05087430628e-07
      60143.71c -2.16346021214e-05
      60145.71c -1.582785376e-05
      95241.71c -1.27411239912e-09
      95243.71c -4.8555127975e-12
      94239.71c -6.67861082287e-05
      94238.71c -3.29987715269e-09
      63154.71c -1.32428865602e-08
      44103.71c -1.64288510922e-06
      92238.71c -0.0626296755441
      54135.71c -3.20976982123e-08
      62149.71c -6.29584479654e-07
      54131.71c -1.00631338803e-05
      1001.71c -0.0171104893331
      62147.71c -9.88330569589e-07
      92234.71c -1.94286866895e-08
      92235.71c -0.014900443065
      92236.71c -0.000125249917864
      43099.71c -1.65734620546e-05
      42095.71c -1.00239310223e-05
      61147.71c -7.56206064747e-06
      40090.71c -0.905024168746
      45103.71c -7.08913186274e-06
      55133.71c -2.40441182411e-05
      93237.71c -3.4376956913e-07
      55135.71c -1.69497854232e-05
      94240.71c -2.18857425291e-06
      94241.71c -9.48932365448e-08
      94242.71c -1.35128281165e-09
m3034 62152.71c -1.8649704322e-06
      62151.71c -1.19329095762e-06
      62150.71c -4.10991828335e-06
      64155.71c -3.1252314605e-09
      63153.71c -7.47584546815e-07
      60143.71c -2.28346436403e-05
      60145.71c -1.67167314671e-05
      95241.71c -1.49486760985e-09
      95243.71c -6.39519819613e-12
      94239.71c -7.03504869305e-05
      94238.71c -3.81995868896e-09
      63154.71c -1.47647491401e-08
      44103.71c -1.73579877775e-06
      92238.71c -0.0626271428882
      54135.71c -3.32817846563e-08
      62149.71c -6.31166517834e-07
      54131.71c -1.06271035382e-05
      1001.71c -0.0171109926207
      62147.71c -1.04331710824e-06
      92234.71c -2.05366373267e-08
      92235.71c -0.0148574104713
      92236.71c -0.000132254023897
      43099.71c -1.75065725246e-05
      42095.71c -1.05912950526e-05
      61147.71c -7.97973761819e-06
      40090.71c -0.905050789112
      45103.71c -7.48823960971e-06
      55133.71c -2.53969831576e-05
      93237.71c -3.77969220039e-07
      55135.71c -1.75897913928e-05
      94240.71c -2.43520588829e-06
      94241.71c -1.11368376384e-07
      94242.71c -1.6796818891e-09
c **  C-4  Fuel ID= 10894
m3041 62152.71c -3.61547363614e-08
      62151.71c -2.90462591255e-08
      62150.71c -8.4534672089e-08
      64155.71c -1.03009317349e-10
      63153.71c -1.55607647361e-08
      60143.71c -4.99077805513e-07
      60145.71c -3.57001010308e-07
      95241.71c -8.81340040556e-12
      95243.71c -1.093280813e-14
      94239.71c -1.33325390527e-06
      94238.71c -3.16040303363e-11
      63154.71c -1.9661391547e-10
      44103.71c -2.95675276587e-08
      92238.71c -0.0626719893497
      54135.71c -6.14473657565e-10
      62149.71c -1.65844264631e-08
      54131.71c -2.29343509958e-07
      1001.71c -0.0171018613778
      62147.71c -2.36428677777e-08
      92234.71c -3.95859394964e-10
      92235.71c -0.0156509348551
      92236.71c -2.78221689559e-06
      43099.71c -3.74731218689e-07
      42095.71c -2.44585753341e-07
      61147.71c -1.72922516916e-07
      40090.71c -0.904567810785
      45103.71c -1.66922274318e-07
      55133.71c -5.45920377388e-07
      93237.71c -5.221594167e-09
      55135.71c -4.25602292425e-07
      94240.71c -2.96757354236e-08
      94241.71c -7.09851073126e-10
      94242.71c -5.98970657876e-12
m3042 62152.71c -4.83063379551e-08
      62151.71c -3.42006138723e-08
      62150.71c -1.11267640785e-07
      64155.71c -1.10055649437e-10
      63153.71c -1.99555453663e-08
      60143.71c -6.29356715932e-07
      60145.71c -4.51048744032e-07
      95241.71c -2.34419631782e-11
      95243.71c -5.51950380263e-14
      94239.71c -1.91527180659e-06
      94238.71c -7.04570156882e-11
      63154.71c -3.30920425309e-10
      44103.71c -3.74552927543e-08
      92238.71c -0.0626715061771
      54135.71c -7.32003717348e-10
      62149.71c -1.67714540973e-08
      54131.71c -2.89741605458e-07
      1001.71c -0.0171019142242
      62147.71c -2.98173875578e-08
      92234.71c -5.53085477901e-10
      92235.71c -0.0156463442367
      92236.71c -3.56751423195e-06
      43099.71c -4.73722217631e-07
      42095.71c -3.0954478736e-07
      61147.71c -2.17684468261e-07
      40090.71c -0.904570605988
      45103.71c -2.11213191192e-07
      55133.71c -6.90066131514e-07
      93237.71c -8.91067919264e-09
      55135.71c -5.08942791997e-07
      94240.71c -5.48553224251e-08
      94241.71c -1.88633698796e-09
      94242.71c -2.04207726597e-11
m3043 62152.71c -6.05270767918e-08
      62151.71c -3.8186235921e-08
      62150.71c -1.36612853206e-07
      64155.71c -1.12909822509e-10
      63153.71c -2.41082534853e-08
      60143.71c -7.50970311851e-07
      60145.71c -5.39207938515e-07
      95241.71c -3.98956665621e-11
      95243.71c -1.36816298727e-13
      94239.71c -2.26687136717e-06
      94238.71c -1.12867098949e-10
      63154.71c -4.72697528966e-10
      44103.71c -4.47808003993e-08
      92238.71c -0.0626712593668
      54135.71c -8.26550322701e-10
      62149.71c -1.66852181585e-08
      54131.71c -3.46289503066e-07
      1001.71c -0.0171019638784
      62147.71c -3.56159865937e-08
      92234.71c -6.56616348372e-10
      92235.71c -0.015642074469
      92236.71c -4.26283467929e-06
      43099.71c -5.66551016465e-07
      42095.71c -3.70672448693e-07
      61147.71c -2.59539112469e-07
      40090.71c -0.904573232343
      45103.71c -2.52646785127e-07
      55133.71c -8.25227926818e-07
      93237.71c -1.20974809831e-08
      55135.71c -5.77162734812e-07
      94240.71c -7.78834210333e-08
      94241.71c -3.20823413488e-09
      94242.71c -4.18959429192e-11
m3044 62152.71c -6.49220022397e-08
      62151.71c -3.94287534843e-08
      62150.71c -1.45449292526e-07
      64155.71c -1.13439551203e-10
      63153.71c -2.55749556689e-08
      60143.71c -7.93143006992e-07
      60145.71c -5.69856892865e-07
      95241.71c -4.69273566424e-11
      95243.71c -1.80143805839e-13
      94239.71c -2.38257034255e-06
      94238.71c -1.30740420468e-10
      63154.71c -5.27550778338e-10
      44103.71c -4.73257185365e-08
      92238.71c -0.0626711725473
      54135.71c -8.56762907025e-10
      62149.71c -1.66479870801e-08
      54131.71c -3.65939549071e-07
      1001.71c -0.0171019812975
      62147.71c -3.76309979337e-08
      92234.71c -6.95041812884e-10
      92235.71c -0.0156405903506
      92236.71c -4.50413429766e-06
      43099.71c -5.98840874185e-07
      42095.71c -3.91976303425e-07
      61147.71c -2.7405223084e-07
      40090.71c -0.904574153692
      45103.71c -2.67061162707e-07
      55133.71c -8.72241647014e-07
      93237.71c -1.33017471703e-08
      55135.71c -5.99174537875e-07
      94240.71c -8.66447292191e-08
      94241.71c -3.77278824761e-09
      94242.71c -5.22423750896e-11
c **  C-5  Fuel ID= 2953
m3051 62152.71c -9.8682469228e-07
      62151.71c -8.30856105188e-07
      62150.71c -2.21611888996e-06
      64155.71c -2.64616829211e-09
      63153.71c -4.32566452675e-07
      60143.71c -1.3615347404e-05
      60145.71c -9.96122514454e-06
      95241.71c -2.62012803144e-10
      95243.71c -3.64144511324e-13
      94239.71c -3.76297179172e-05
      94238.71c -8.5304944262e-10
      63154.71c -5.12701428226e-09
      44103.71c -1.07544017251e-06
      92238.71c -0.0626519875462
      54135.71c -2.3760809765e-08
      62149.71c -5.95592437248e-07
      54131.71c -6.32222211624e-06
      1001.71c -0.0171071852073
      62147.71c -6.08055752936e-07
      92234.71c -1.11182944968e-08
      92235.71c -0.0151856458237
      92236.71c -7.77754185908e-05
      43099.71c -1.04131751352e-05
      42095.71c -6.15012940397e-06
      61147.71c -4.78925306851e-06
      40090.71c -0.904849403804
      45103.71c -4.40878768121e-06
      55133.71c -1.50972157444e-05
      93237.71c -1.40369455272e-07
      55135.71c -1.18845365428e-05
      94240.71c -7.81211342098e-07
      94241.71c -1.96070781616e-08
      94242.71c -1.79586415596e-10
m3052 62152.71c -1.32102003451e-06
      62151.71c -9.88908345303e-07
      62150.71c -2.95479419737e-06
      64155.71c -2.84131079827e-09
      63153.71c -5.56780956448e-07
      60143.71c -1.72440953731e-05
      60145.71c -1.2640274695e-05
      95241.71c -7.01584900861e-10
      95243.71c -1.86774959571e-12
      94239.71c -5.42204355921e-05
      94238.71c -1.9314350902e-09
      63154.71c -8.69438133207e-09
      44103.71c -1.36882559261e-06
      92238.71c -0.0626382913686
      54135.71c -2.84532133042e-08
      62149.71c -6.20724011485e-07
      54131.71c -8.02169416947e-06
      1001.71c -0.0171086978822
      62147.71c -7.69671782629e-07
      92234.71c -1.57563144686e-08
      92235.71c -0.0150547887757
      92236.71c -0.000100179245747
      43099.71c -1.32207557301e-05
      42095.71c -7.81386619368e-06
      61147.71c -6.05661278225e-06
      40090.71c -0.904929413636
      45103.71c -5.60156854675e-06
      55133.71c -1.91657014673e-05
      93237.71c -2.40233142589e-07
      55135.71c -1.42611821671e-05
      94240.71c -1.45048231012e-06
      94241.71c -5.24655463193e-08
      94242.71c -6.19125203192e-10
m3053 62152.71c -1.65144556964e-06
      62151.71c -1.11357896242e-06
      62150.71c -3.65364981178e-06
      64155.71c -2.93158155534e-09
      63153.71c -6.72735579968e-07
      60143.71c -2.05884047381e-05
      60145.71c -1.51195074155e-05
      95241.71c -1.19590565442e-09
      95243.71c -4.61699356234e-12
      94239.71c -6.42289279692e-05
      94238.71c -3.10491470412e-09
      63154.71c -1.24109128642e-08
      44103.71c -1.63910379894e-06
      92238.71c -0.0626312244253
      54135.71c -3.22170843826e-08
      62149.71c -6.29478143219e-07
      54131.71c -9.59269207986e-06
      1001.71c -0.0171101012289
      62147.71c -9.19402295589e-07
      92234.71c -1.8723382631e-08
      92235.71c -0.0149346768904
      92236.71c -0.000119780820788
      43099.71c -1.58200522214e-05
      42095.71c -9.35832315425e-06
      61147.71c -7.22635950777e-06
      40090.71c -0.905003640774
      45103.71c -6.70283190554e-06
      55133.71c -2.2931765608e-05
      93237.71c -3.25684366908e-07
      55135.71c -1.61820979942e-05
      94240.71c -2.05860213476e-06
      94241.71c -8.93563761757e-08
      94242.71c -1.27300677645e-09
m3054 62152.71c -1.76959055308e-06
      62151.71c -1.15283937393e-06
      62150.71c -3.89708129907e-06
      64155.71c -2.95072965398e-09
      63153.71c -7.13466776182e-07
      60143.71c -2.17423884218e-05
      60145.71c -1.59771782902e-05
      95241.71c -1.40008582627e-09
      95243.71c -6.09340155257e-12
      94239.71c -6.75432868535e-05
      94238.71c -3.57747951862e-09
      63154.71c -1.38465061717e-08
      44103.71c -1.73264928591e-06
      92238.71c -0.06262888372
      54135.71c -3.34166922357e-08
      62149.71c -6.31297675904e-07
      54131.71c -1.01359173167e-05
      1001.71c -0.0171105881785
      62147.71c -9.71067352849e-07
      92234.71c -1.96909436278e-08
      92235.71c -0.0148931179109
      92236.71c -0.000126559540609
      43099.71c -1.67197307127e-05
      42095.71c -9.89328268831e-06
      61147.71c -7.63011419204e-06
      40090.71c -0.905029396974
      45103.71c -7.08398884937e-06
      55133.71c -2.42349665879e-05
      93237.71c -3.55828271842e-07
      55135.71c -1.67990566864e-05
      94240.71c -2.28889825387e-06
      94241.71c -1.0457872684e-07
      94242.71c -1.57963653365e-09
c **  C-6  Fuel ID= 3082
m3061 62152.71c -2.58848391008e-06
      62151.71c -1.49230770068e-06
      62150.71c -5.72508873055e-06
      64155.71c -6.43731174501e-09
      63153.71c -1.00727976245e-06
      60143.71c -3.16170720456e-05
      60145.71c -2.230557466e-05
      95241.71c -6.19648172766e-09
      95243.71c -4.79450277022e-11
      94239.71c -7.96427803163e-05
      94238.71c -6.48373477462e-09
      63154.71c -2.13140551928e-08
      44103.71c -1.18124878824e-06
      92238.71c -0.0626264126555
      54135.71c -2.41042114675e-08
      62149.71c -6.24820678839e-07
      54131.71c -1.44767818129e-05
      1001.71c -0.0171139302703
      62147.71c -2.42317474698e-06
      92234.71c -2.40769168667e-08
      92235.71c -0.0145893016372
      92236.71c -0.000173219918379
      43099.71c -2.35287808027e-05
      42095.71c -1.84733461713e-05
      61147.71c -9.97585351757e-06
      40090.71c -0.905206170049
      45103.71c -1.10858826451e-05
      55133.71c -3.44244818013e-05
      93237.71c -4.89348007041e-07
      55135.71c -2.64657068734e-05
      94240.71c -3.18117848363e-06
      94241.71c -1.6253105761e-07
      94242.71c -5.08680607767e-09
m3062 62152.71c -3.38887529205e-06
      62151.71c -1.68233623397e-06
      62150.71c -7.27912252209e-06
      64155.71c -6.47275690417e-09
      63153.71c -1.28674049196e-06
      60143.71c -3.91078298007e-05
      60145.71c -2.76877560397e-05
      95241.71c -1.44032840595e-08
      95243.71c -2.02654792081e-10
      94239.71c -0.000111723738282
      94238.71c -1.46732375999e-08
      63154.71c -3.47234209203e-08
      44103.71c -1.47365043405e-06
      92238.71c -0.062596904607
      54135.71c -2.82574766907e-08
      62149.71c -6.29107642249e-07
      54131.71c -1.79613889051e-05
      1001.71c -0.0171169606178
      62147.71c -2.99135570291e-06
      92234.71c -3.35141893902e-08
      92235.71c -0.0143271673486
      92236.71c -0.000218078110347
      43099.71c -2.92314268553e-05
      42095.71c -2.29714092526e-05
      61147.71c -1.23146131244e-05
      40090.71c -0.905366454056
      45103.71c -1.37817870622e-05
      55133.71c -4.27603188552e-05
      93237.71c -8.41044004543e-07
      55135.71c -3.12015388005e-05
      94240.71c -5.58316324831e-06
      94241.71c -3.90734784443e-07
      94242.71c -1.50760864568e-08
m3063 62152.71c -4.20812573124e-06
      62151.71c -1.81663963431e-06
      62150.71c -8.79052051162e-06
      64155.71c -6.35315741605e-09
      63153.71c -1.55947395849e-06
      60143.71c -4.62744120286e-05
      60145.71c -3.2879304105e-05
      95241.71c -2.25588239632e-08
      95243.71c -4.52278846881e-10
      94239.71c -0.000130415228912
      94238.71c -2.37663454095e-08
      63154.71c -4.88499492829e-08
      44103.71c -1.74997319503e-06
      92238.71c -0.0625814401027
      54135.71c -3.16169001825e-08
      62149.71c -6.2296151033e-07
      54131.71c -2.13172674821e-05
      1001.71c -0.017119898198
      62147.71c -3.5409422612e-06
      92234.71c -3.93655217584e-08
      92235.71c -0.0140762113052
      92236.71c -0.000258744597319
      43099.71c -3.47361558662e-05
      42095.71c -2.73337577866e-05
      61147.71c -1.45644973615e-05
      40090.71c -0.905521831321
      45103.71c -1.63717595365e-05
      55133.71c -5.0805083256e-05
      93237.71c -1.14773149175e-06
      55135.71c -3.51760225967e-05
      94240.71c -7.73352944002e-06
      94241.71c -6.29526036169e-07
      94242.71c -2.86002214496e-08
m3064 62152.71c -4.4951135138e-06
      62151.71c -1.85496504407e-06
      62150.71c -9.3085547381e-06
      64155.71c -6.2945960572e-09
      63153.71c -1.65509565726e-06
      60143.71c -4.87059194828e-05
      60145.71c -3.46514214398e-05
      95241.71c -2.57477870338e-08
      95243.71c -5.753966856e-10
      94239.71c -0.000136456665208
      94238.71c -2.75355442932e-08
      63154.71c -5.41954763571e-08
      44103.71c -1.84571589244e-06
      92238.71c -0.0625762896009
      54135.71c -3.2684947115e-08
      62149.71c -6.20732673543e-07
      54131.71c -2.24614192201e-05
      1001.71c -0.0171209046456
      62147.71c -3.72743900492e-06
      92234.71c -4.12984086856e-08
      92235.71c -0.0139906157625
      92236.71c -0.000272589409423
      43099.71c -3.66165904663e-05
      42095.71c -2.88226995415e-05
      61147.71c -1.53278652161e-05
      40090.71c -0.905575065297
      45103.71c -1.72546716857e-05
      55133.71c -5.35523166203e-05
      93237.71c -1.26270727399e-06
      55135.71c -3.64336301529e-05
      94240.71c -8.53291937691e-06
      94241.71c -7.25853944001e-07
      94242.71c -3.46566114507e-08
c **  C-8  Fuel ID= 2933
m3081 62152.71c -7.64676382837e-07
      62151.71c -6.90327520631e-07
      62150.71c -1.66566654844e-06
      64155.71c -1.93689641076e-09
      63153.71c -3.4443355908e-07
      60143.71c -1.06819041462e-05
      60145.71c -7.9818129961e-06
      95241.71c -1.67774447353e-10
      95243.71c -2.33088450507e-13
      94239.71c -2.99729837933e-05
      94238.71c -5.67532615564e-10
      63154.71c -3.57123507867e-09
      44103.71c -1.05815341989e-06
      92238.71c -0.0626565035733
      54135.71c -2.39722146977e-08
      62149.71c -5.79417214155e-07
      54131.71c -5.00813394324e-06
      1001.71c -0.0171061091109
      62147.71c -4.1847981959e-07
      92234.71c -8.85343948037e-09
      92235.71c -0.0152814512984
      92236.71c -6.22837017482e-05
      43099.71c -8.31092704837e-06
      42095.71c -4.322149235e-06
      61147.71c -3.84300200998e-06
      40090.71c -0.904792485896
      45103.71c -3.33701682575e-06
      55133.71c -1.19943058118e-05
      93237.71c -1.0189462452e-07
      55135.71c -9.4989761557e-06
      94240.71c -5.40177244518e-07
      94241.71c -1.27958907536e-08
      94242.71c -1.15837089109e-10
m3082 62152.71c -1.00845015042e-06
      62151.71c -8.23734669412e-07
      62150.71c -2.21754203648e-06
      64155.71c -2.10245349644e-09
      63153.71c -4.38358090056e-07
      60143.71c -1.34119218117e-05
      60145.71c -1.00377784898e-05
      95241.71c -4.38053744685e-10
      95243.71c -1.14068402194e-12
      94239.71c -4.29961129891e-05
      94238.71c -1.25365919595e-09
      63154.71c -5.94919441949e-09
      44103.71c -1.33418453012e-06
      92238.71c -0.062645869763
      54135.71c -2.85296510003e-08
      62149.71c -6.11161028759e-07
      54131.71c -6.2978511059e-06
      1001.71c -0.01710727432
      62147.71c -5.24910088067e-07
      92234.71c -1.24788623863e-08
      92235.71c -0.0151809230837
      92236.71c -7.95407258535e-05
      43099.71c -1.04564382534e-05
      42095.71c -5.4405605518e-06
      61147.71c -4.81921912714e-06
      40090.71c -0.904854117237
      45103.71c -4.20135134388e-06
      55133.71c -1.50893595371e-05
      93237.71c -1.70409661464e-07
      55135.71c -1.13224359061e-05
      94240.71c -9.88562572363e-07
      94241.71c -3.33900384783e-08
      94242.71c -3.8570889254e-10
m3083 62152.71c -1.25483062492e-06
      62151.71c -9.35136763603e-07
      62150.71c -2.75836485549e-06
      64155.71c -2.19661998515e-09
      63153.71c -5.28106517191e-07
      60143.71c -1.59955140691e-05
      60145.71c -1.19901594959e-05
      95241.71c -7.4421467178e-10
      95243.71c -2.81275198277e-12
      94239.71c -5.10407275018e-05
      94238.71c -2.0020764805e-09
      63154.71c -8.47353688396e-09
      44103.71c -1.59539251885e-06
      92238.71c -0.0626404409066
      54135.71c -3.22925382652e-08
      62149.71c -6.24875600059e-07
      54131.71c -7.52118986686e-06
      1001.71c -0.0171083816055
      62147.71c -6.25940951026e-07
      92234.71c -1.47931441677e-08
      92235.71c -0.0150862485268
      92236.71c -9.49933866439e-05
      43099.71c -1.24944449058e-05
      42095.71c -6.50476230673e-06
      61147.71c -5.74453428385e-06
      40090.71c -0.904912684828
      45103.71c -5.02009113881e-06
      55133.71c -1.80288975803e-05
      93237.71c -2.28997086372e-07
      55135.71c -1.28384032278e-05
      94240.71c -1.40233222754e-06
      94241.71c -5.67504426796e-08
      94242.71c -7.89536390152e-10
m3084 62152.71c -1.34239428641e-06
      62151.71c -9.70574297593e-07
      62150.71c -2.94611605394e-06
      64155.71c -2.21975515771e-09
      63153.71c -5.59210534237e-07
      60143.71c -1.68804646887e-05
      60145.71c -1.26608662869e-05
      95241.71c -8.71550171796e-10
      95243.71c -3.67401526928e-12
      94239.71c -5.36460227296e-05
      94238.71c -2.30947767255e-09
      63154.71c -9.4377158956e-09
      44103.71c -1.68540020254e-06
      92238.71c -0.0626386268652
      54135.71c -3.34965819483e-08
      62149.71c -6.27946833929e-07
      54131.71c -7.9411249781e-06
      1001.71c -0.0171087643997
      62147.71c -6.60523392798e-07
      92234.71c -1.55929514581e-08
      92235.71c -0.0150537496743
      92236.71c -0.000100278697242
      43099.71c -1.31947566875e-05
      42095.71c -6.87052275175e-06
      61147.71c -6.06149600533e-06
      40090.71c -0.904932931943
      45103.71c -5.30123593997e-06
      55133.71c -1.90386903829e-05
      93237.71c -2.50660806148e-07
      55135.71c -1.33223058704e-05
      94240.71c -1.55674909761e-06
      94241.71c -6.64471515479e-08
      94242.71c -9.79454084308e-10
c **  C-9  Fuel ID= 6314
m3091 62152.71c -1.11171670695e-06
      62151.71c -9.47415695119e-07
      62150.71c -2.56045528394e-06
      64155.71c -3.0280162355e-09
      63153.71c -4.89431792766e-07
      60143.71c -1.55656359417e-05
      60145.71c -1.12824125288e-05
      95241.71c -2.30001519867e-10
      95243.71c -2.90344337105e-13
      94239.71c -4.21728350219e-05
      94238.71c -8.65764040446e-10
      63154.71c -5.58670148128e-09
      44103.71c -1.13645105019e-06
      92238.71c -0.062649806522
      54135.71c -2.39183423942e-08
      62149.71c -6.27782057978e-07
      54131.71c -7.19517265373e-06
      1001.71c -0.017107909788
      62147.71c -6.60602387282e-07
      92234.71c -1.24307130842e-08
      92235.71c -0.0151218515031
      92236.71c -8.79212058051e-05
      43099.71c -1.18120669919e-05
      42095.71c -7.0068509565e-06
      61147.71c -5.49440371501e-06
      40090.71c -0.90488772901
      45103.71c -5.07413802521e-06
      55133.71c -1.71578996059e-05
      93237.71c -1.52990027241e-07
      55135.71c -1.34266020973e-05
      94240.71c -8.41783322834e-07
      94241.71c -1.91065146657e-08
      94242.71c -1.59100569948e-10
m3092 62152.71c -1.46877666166e-06
      62151.71c -1.1195465761e-06
      62150.71c -3.36802902437e-06
      64155.71c -3.26870356627e-09
      63153.71c -6.2234361068e-07
      60143.71c -1.95059164543e-05
      60145.71c -1.41614198781e-05
      95241.71c -6.01160536694e-10
      95243.71c -1.41947236275e-12
      94239.71c -6.02641101084e-05
      94238.71c -1.90194440157e-09
      63154.71c -9.28001682677e-09
      44103.71c -1.42971423442e-06
      92238.71c -0.0626349821689
      54135.71c -2.83709356194e-08
      62149.71c -6.41793463884e-07
      54131.71c -9.03100703974e-06
      1001.71c -0.0171095310856
      62147.71c -8.27805096981e-07
      92234.71c -1.75540341414e-08
      92235.71c -0.0149811918299
      92236.71c -0.000112045648449
      43099.71c -1.48337757903e-05
      42095.71c -8.80888346852e-06
      61147.71c -6.87475654129e-06
      40090.71c -0.904973484217
      45103.71c -6.37799738719e-06
      55133.71c -2.15455696183e-05
      93237.71c -2.57433683074e-07
      55135.71c -1.59786684195e-05
      94240.71c -1.5361140459e-06
      94241.71c -4.98820136155e-08
      94242.71c -5.28648935314e-10
m3093 62152.71c -1.83262771563e-06
      62151.71c -1.25934517259e-06
      62150.71c -4.15083233781e-06
      64155.71c -3.39194738627e-09
      63153.71c -7.49715591402e-07
      60143.71c -2.32456016497e-05
      60145.71c -1.69040507083e-05
      95241.71c -1.02091376032e-09
      95243.71c -3.50092053071e-12
      94239.71c -7.14746167091e-05
      94238.71c -3.01730446527e-09
      63154.71c -1.32043252146e-08
      44103.71c -1.7075864167e-06
      92238.71c -0.0626270983032
      54135.71c -3.20383273201e-08
      62149.71c -6.42143975721e-07
      54131.71c -1.07779424665e-05
      1001.71c -0.0171110821184
      62147.71c -9.87129966774e-07
      92234.71c -2.06672293048e-08
      92235.71c -0.0148482856684
      92236.71c -0.000133713530573
      43099.71c -1.7713427416e-05
      42095.71c -1.05296817178e-05
      61147.71c -8.18727718323e-06
      40090.71c -0.905055522907
      45103.71c -7.61723350682e-06
      55133.71c -2.57261793008e-05
      93237.71c -3.46340503193e-07
      55135.71c -1.81056445483e-05
      94240.71c -2.1810243692e-06
      94241.71c -8.46475231875e-08
      94242.71c -1.07991262891e-09
m3094 62152.71c -1.96380081805e-06
      62151.71c -1.30365920095e-06
      62150.71c -4.42449040779e-06
      64155.71c -3.42011442899e-09
      63153.71c -7.94607121765e-07
      60143.71c -2.45452035495e-05
      60145.71c -1.78591742011e-05
      95241.71c -1.19899142873e-09
      95243.71c -4.57881628622e-12
      94239.71c -7.51580002882e-05
      94238.71c -3.47877362194e-09
      63154.71c -1.47206680763e-08
      44103.71c -1.80406326084e-06
      92238.71c -0.0626245343373
      54135.71c -3.32125032335e-08
      62149.71c -6.41720014493e-07
      54131.71c -1.13863597068e-05
      1001.71c -0.0171116235074
      62147.71c -1.04267277005e-06
      92234.71c -2.17914624708e-08
      92235.71c -0.0148020411043
      92236.71c -0.000141219937215
      43099.71c -1.87167695665e-05
      42095.71c -1.11310946433e-05
      61147.71c -8.64293330515e-06
      40090.71c -0.905084158565
      45103.71c -8.04932459049e-06
      55133.71c -2.7182673814e-05
      93237.71c -3.79256643548e-07
      55135.71c -1.87937916906e-05
      94240.71c -2.42446997891e-06
      94241.71c -9.93128901247e-08
      94242.71c -1.34279647561e-09
c **  C-10  Fuel ID= 6527
m3101 62152.71c -1.09401552831e-06
      62151.71c -9.37049015794e-07
      62150.71c -2.51709869001e-06
      64155.71c -2.97355737456e-09
      63153.71c -4.82601482799e-07
      60143.71c -1.53399665791e-05
      60145.71c -1.11292502098e-05
      95241.71c -2.22592080379e-10
      95243.71c -2.80463821211e-13
      94239.71c -4.1610554345e-05
      94238.71c -8.42607566697e-10
      63154.71c -5.45782152605e-09
      44103.71c -1.1353178893e-06
      92238.71c -0.0626501308161
      54135.71c -2.39173911689e-08
      62149.71c -6.27313021357e-07
      54131.71c -7.09378475798e-06
      1001.71c -0.0171078264406
      62147.71c -6.45118266649e-07
      92234.71c -1.22631186569e-08
      92235.71c -0.0151292577163
      92236.71c -8.67306053923e-05
      43099.71c -1.16495234983e-05
      42095.71c -6.85964596381e-06
      61147.71c -5.42237368576e-06
      40090.71c -0.904883320522
      45103.71c -4.99102043674e-06
      55133.71c -1.6918248278e-05
      93237.71c -1.49982173825e-07
      55135.71c -1.32443511915e-05
      94240.71c -8.22300362555e-07
      94241.71c -1.85526672539e-08
      94242.71c -1.53958320157e-10
m3102 62152.71c -1.44514917684e-06
      62151.71c -1.10802657144e-06
      62150.71c -3.31319269782e-06
      64155.71c -3.21336285482e-09
      63153.71c -6.13608786775e-07
      60143.71c -1.92245561639e-05
      60145.71c -1.3970010318e-05
      95241.71c -5.81823647718e-10
      95243.71c -1.37120291298e-12
      94239.71c -5.94715625194e-05
      94238.71c -1.85037825839e-09
      63154.71c -9.06662836695e-09
      44103.71c -1.42839573769e-06
      92238.71c -0.062635512211
      54135.71c -2.83736377086e-08
      62149.71c -6.41691742628e-07
      54131.71c -8.90429809871e-06
      1001.71c -0.0171094264794
      62147.71c -8.08436458077e-07
      92234.71c -1.73184977294e-08
      92235.71c -0.014990460444
      92236.71c -0.000110536213719
      43099.71c -1.46304583907e-05
      42095.71c -8.62409215054e-06
      61147.71c -6.78526190248e-06
      40090.71c -0.904967951288
      45103.71c -6.27381299501e-06
      55133.71c -2.12458112251e-05
      93237.71c -2.52216931377e-07
      55135.71c -1.57626229616e-05
      94240.71c -1.50080045274e-06
      94241.71c -4.84409490787e-08
      94242.71c -5.1158801412e-10
m3103 62152.71c -1.80293632365e-06
      62151.71c -1.24718173596e-06
      62150.71c -4.08522216568e-06
      64155.71c -3.33777203319e-09
      63153.71c -7.39148268346e-07
      60143.71c -2.29117548091e-05
      60145.71c -1.66763530108e-05
      95241.71c -9.88117272782e-10
      95243.71c -3.38190361074e-12
      94239.71c -7.05468168185e-05
      94238.71c -2.93477141632e-09
      63154.71c -1.29014919083e-08
      44103.71c -1.70612937986e-06
      92238.71c -0.0626277388965
      54135.71c -3.20451862066e-08
      62149.71c -6.42255490207e-07
      54131.71c -1.06272447534e-05
      1001.71c -0.0171109572019
      62147.71c -9.64065625935e-07
      92234.71c -2.03918518055e-08
      92235.71c -0.0148593078186
      92236.71c -0.000131919214509
      43099.71c -1.74713970568e-05
      42095.71c -1.03090614638e-05
      61147.71c -8.08126841408e-06
      40090.71c -0.905048915705
      45103.71c -7.49310872105e-06
      55133.71c -2.53693664561e-05
      93237.71c -3.39189031304e-07
      55135.71c -1.78616758018e-05
      94240.71c -2.13113124509e-06
      94241.71c -8.22090767768e-08
      94242.71c -1.0450991351e-09
m3104 62152.71c -1.93194265135e-06
      62151.71c -1.29135682121e-06
      62150.71c -4.35518896649e-06
      64155.71c -3.36657175052e-09
      63153.71c -7.8339492887e-07
      60143.71c -2.41933633075e-05
      60145.71c -1.76190058074e-05
      95241.71c -1.16050408119e-09
      95243.71c -4.42328361141e-12
      94239.71c -7.41872271243e-05
      94238.71c -3.38344713056e-09
      63154.71c -1.43834805916e-08
      44103.71c -1.80259532418e-06
      92238.71c -0.0626252087638
      54135.71c -3.32213373296e-08
      62149.71c -6.41879688787e-07
      54131.71c -1.12274067407e-05
      1001.71c -0.0171114916272
      62147.71c -1.01832434073e-06
      92234.71c -2.15018906814e-08
      92235.71c -0.0148136650688
      92236.71c -0.000139328343907
      43099.71c -1.84614169329e-05
      42095.71c -1.08980029716e-05
      61147.71c -8.53128902539e-06
      40090.71c -0.905077183036
      45103.71c -7.91830557433e-06
      55133.71c -2.68062261369e-05
      93237.71c -3.71399952507e-07
      55135.71c -1.8540931329e-05
      94240.71c -2.36912405828e-06
      94241.71c -9.64573436954e-08
      94242.71c -1.29954140965e-09
c **  C-11  Fuel ID= 6317
m3111 62152.71c -1.17847726851e-06
      62151.71c -9.86513957864e-07
      62150.71c -2.7239760651e-06
      64155.71c -3.23340957186e-09
      63153.71c -5.15192525335e-07
      60143.71c -1.6416755163e-05
      60145.71c -1.18600691151e-05
      95241.71c -2.57946457951e-10
      95243.71c -3.27609018918e-13
      94239.71c -4.42934945846e-05
      94238.71c -9.53099422057e-10
      63154.71c -6.07277635579e-09
      44103.71c -1.14072480273e-06
      92238.71c -0.0626485834366
      54135.71c -2.39219299708e-08
      62149.71c -6.295510446e-07
      54131.71c -7.57756034636e-06
      1001.71c -0.0171082241352
      62147.71c -7.19001243648e-07
      92234.71c -1.30628008286e-08
      92235.71c -0.0150939187332
      92236.71c -9.2411593354e-05
      43099.71c -1.24251049784e-05
      42095.71c -7.56203929608e-06
      61147.71c -5.76606727483e-06
      40090.71c -0.904904355762
      45103.71c -5.38761866939e-06
      55133.71c -1.80617522562e-05
      93237.71c -1.64334242635e-07
      55135.71c -1.41139672196e-05
      94240.71c -9.15263930864e-07
      94241.71c -2.1195367899e-08
      94242.71c -1.78494729555e-10
m3112 62152.71c -1.55788847659e-06
      62151.71c -1.16299464224e-06
      62150.71c -3.57484598437e-06
      64155.71c -3.47742283011e-09
      63153.71c -6.55287278496e-07
      60143.71c -2.05670757913e-05
      60145.71c -1.48833271581e-05
      95241.71c -6.74090232129e-10
      95243.71c -1.60152213689e-12
      94239.71c -6.3253228697e-05
      94238.71c -2.09642776034e-09
      63154.71c -1.00848182301e-08
      44103.71c -1.43468698701e-06
      92238.71c -0.0626329830983
      54135.71c -2.83607446037e-08
      62149.71c -6.42177108855e-07
      54131.71c -9.50889386477e-06
      1001.71c -0.0171099256112
      62147.71c -9.00854538285e-07
      92234.71c -1.84423672548e-08
      92235.71c -0.0149462349556
      92236.71c -0.00011773852983
      43099.71c -1.5600593868e-05
      42095.71c -9.50582983647e-06
      61147.71c -7.21228843482e-06
      40090.71c -0.904994351835
      45103.71c -6.77093215553e-06
      55133.71c -2.2676117989e-05
      93237.71c -2.77108829005e-07
      55135.71c -1.6793490776e-05
      94240.71c -1.66930039232e-06
      94241.71c -5.53170346213e-08
      94242.71c -5.92994745443e-10
m3113 62152.71c -1.94460975044e-06
      62151.71c -1.30521996302e-06
      62150.71c -4.39828320114e-06
      64155.71c -3.59627146366e-09
      63153.71c -7.89570588401e-07
      60143.71c -2.45047156902e-05
      60145.71c -1.77628198596e-05
      95241.71c -1.14460676237e-09
      95243.71c -3.94979665217e-12
      94239.71c -7.49738436807e-05
      94238.71c -3.32858049992e-09
      63154.71c -1.43464707182e-08
      44103.71c -1.71308167777e-06
      92238.71c -0.0626246822853
      54135.71c -3.20124588102e-08
      62149.71c -6.41723395271e-07
      54131.71c -1.13463037129e-05
      1001.71c -0.017111553245
      62147.71c -1.07411786616e-06
      92234.71c -2.17058243536e-08
      92235.71c -0.0148067152764
      92236.71c -0.000140480851015
      43099.71c -1.86262526591e-05
      42095.71c -1.13617580644e-05
      61147.71c -8.5870926477e-06
      40090.71c -0.905080442181
      45103.71c -8.0853741076e-06
      55133.71c -2.70719103603e-05
      93237.71c -3.73312508228e-07
      55135.71c -1.9025780481e-05
      94240.71c -2.36919788303e-06
      94241.71c -9.38442018406e-08
      94242.71c -1.21121283465e-09
m3114 62152.71c -2.083954913e-06
      62151.71c -1.35005801971e-06
      62150.71c -4.68586301087e-06
      64155.71c -3.62205235387e-09
      63153.71c -8.36894265991e-07
      60143.71c -2.58721802789e-05
      60145.71c -1.87649769817e-05
      95241.71c -1.34414769136e-09
      95243.71c -5.16541274541e-12
      94239.71c -7.88193023347e-05
      94238.71c -3.83830053408e-09
      63154.71c -1.5992381463e-08
      44103.71c -1.80959963095e-06
      92238.71c -0.0626219907163
      54135.71c -3.31791851576e-08
      62149.71c -6.41117797787e-07
      54131.71c -1.19858559041e-05
      1001.71c -0.0171121208976
      62147.71c -1.13450365019e-06
      92234.71c -2.28835917393e-08
      92235.71c -0.0147582009506
      92236.71c -0.000148354141897
      43099.71c -1.96798401937e-05
      42095.71c -1.20102073405e-05
      61147.71c -9.06400327821e-06
      40090.71c -0.905110466997
      45103.71c -8.54346700175e-06
      55133.71c -2.86024582724e-05
      93237.71c -4.08888405174e-07
      55135.71c -1.97474626346e-05
      94240.71c -2.63320888931e-06
      94241.71c -1.10082674843e-07
      94242.71c -1.50593434076e-09
c **  C-12  Fuel ID= 6526
m3121 62152.71c -1.02657943476e-06
      62151.71c -8.97555128346e-07
      62150.71c -2.35192328646e-06
      64155.71c -2.7660857188e-09
      63153.71c -4.56580084372e-07
      60143.71c -1.44802351135e-05
      60145.71c -1.05457484733e-05
      95241.71c -1.94364375055e-10
      95243.71c -2.42822068202e-13
      94239.71c -3.94684364041e-05
      94238.71c -7.54388462075e-10
      63154.71c -4.96682819187e-09
      44103.71c -1.13100089181e-06
      92238.71c -0.0626513662775
      54135.71c -2.39137672907e-08
      62149.71c -6.25526134839e-07
      54131.71c -6.70752778813e-06
      1001.71c -0.0171075089126
      62147.71c -5.86128488114e-07
      92234.71c -1.16246349882e-08
      92235.71c -0.0151574731304
      92236.71c -8.21947808312e-05
      43099.71c -1.10302823469e-05
      42095.71c -6.29883982418e-06
      61147.71c -5.14796123614e-06
      40090.71c -0.904866525529
      45103.71c -4.67436776705e-06
      55133.71c -1.60052497882e-05
      93237.71c -1.38523169396e-07
      55135.71c -1.25500308082e-05
      94240.71c -7.48076224206e-07
      94241.71c -1.64426774819e-08
      94242.71c -1.34367916315e-10
m3122 62152.71c -1.35513566366e-06
      62151.71c -1.06413886612e-06
      62150.71c -3.10428301316e-06
      64155.71c -3.00253161743e-09
      63153.71c -5.8033177093e-07
      60143.71c -1.81526592437e-05
      60145.71c -1.32407982542e-05
      95241.71c -5.08155996499e-10
      95243.71c -1.18731102662e-12
      94239.71c -5.64521978515e-05
      94238.71c -1.65392697532e-09
      63154.71c -8.25368339673e-09
      44103.71c -1.42337266716e-06
      92238.71c -0.0626375315096
      54135.71c -2.83839318444e-08
      62149.71c -6.41304215658e-07
      54131.71c -8.42157566667e-06
      1001.71c -0.0171090279618
      62147.71c -7.34647849313e-07
      92234.71c -1.64211758145e-08
      92235.71c -0.0150257710374
      92236.71c -0.000104785727617
      43099.71c -1.38558810891e-05
      42095.71c -7.92009357155e-06
      61147.71c -6.44431461538e-06
      40090.71c -0.904946872517
      45103.71c -5.87690222645e-06
      55133.71c -2.01038231425e-05
      93237.71c -2.32342697987e-07
      55135.71c -1.49395556389e-05
      94240.71c -1.36626642984e-06
      94241.71c -4.29509325711e-08
      94242.71c -4.46591106211e-10
m3123 62152.71c -1.68982117304e-06
      62151.71c -1.20084275098e-06
      62150.71c -3.83526741426e-06
      64155.71c -3.13138045594e-09
      63153.71c -6.98889989455e-07
      60143.71c -2.16399001355e-05
      60145.71c -1.58088942077e-05
      95241.71c -8.63172654838e-10
      95243.71c -2.9284854337e-12
      94239.71c -6.70121821194e-05
      94238.71c -2.620345664e-09
      63154.71c -1.17477893445e-08
      44103.71c -1.70057851373e-06
      92238.71c -0.0626301793615
      54135.71c -3.2071316473e-08
      62149.71c -6.42680326396e-07
      54131.71c -1.00531324139e-05
      1001.71c -0.0171104813081
      62147.71c -8.76197519612e-07
      92234.71c -1.93427474954e-08
      92235.71c -0.0149012988501
      92236.71c -0.000125083417389
      43099.71c -1.65493351829e-05
      42095.71c -9.46856556255e-06
      61147.71c -7.67740732549e-06
      40090.71c -0.90502374428
      45103.71c -7.02023113265e-06
      55133.71c -2.40100183095e-05
      93237.71c -3.11944103875e-07
      55135.71c -1.69322292634e-05
      94240.71c -1.94105365456e-06
      94241.71c -7.29193394297e-08
      94242.71c -9.12470338232e-10
m3124 62152.71c -1.81057274974e-06
      62151.71c -1.2444885054e-06
      62150.71c -4.09117160502e-06
      64155.71c -3.1625904706e-09
      63153.71c -7.40679892515e-07
      60143.71c -2.28529592609e-05
      60145.71c -1.67040374544e-05
      95241.71c -1.01387902175e-09
      95243.71c -3.83075154194e-12
      94239.71c -7.04888773567e-05
      94238.71c -3.02028226322e-09
      63154.71c -1.30988990825e-08
      44103.71c -1.79700293305e-06
      92238.71c -0.062627778123
      54135.71c -3.32549925421e-08
      62149.71c -6.4248799916e-07
      54131.71c -1.06218444042e-05
      1001.71c -0.0171109892041
      62147.71c -9.25564248846e-07
      92234.71c -2.03987104534e-08
      92235.71c -0.0148579488291
      92236.71c -0.000132121950145
      43099.71c -1.74886012555e-05
      42095.71c -1.00099947718e-05
      61147.71c -8.10595835961e-06
      40090.71c -0.905050608396
      45103.71c -7.41916307019e-06
      55133.71c -2.53720752651e-05
      93237.71c -3.41468355131e-07
      55135.71c -1.75776104477e-05
      94240.71c -2.15827297554e-06
      94241.71c -8.55785824503e-08
      94242.71c -1.13475279674e-09
c **  D-1  Fuel ID= 3380
m4011 62152.71c -3.44117370692e-06
      62151.71c -1.55058447535e-06
      62150.71c -7.17629051923e-06
      64155.71c -7.12541778113e-09
      63153.71c -1.29770764526e-06
      60143.71c -3.85738927034e-05
      60145.71c -2.73006965535e-05
      95241.71c -2.91055112172e-08
      95243.71c -2.81563686739e-10
      94239.71c -9.30032672405e-05
      94238.71c -2.03645641842e-08
      63154.71c -3.90084287913e-08
      44103.71c -1.17923064695e-06
      92238.71c -0.0626144286705
      54135.71c -2.39266791423e-08
      62149.71c -6.16885478453e-07
      54131.71c -1.7753265751e-05
      1001.71c -0.0171167143106
      62147.71c -4.07076688304e-06
      92234.71c -2.91922766748e-08
      92235.71c -0.0143484199023
      92236.71c -0.000211484947356
      43099.71c -2.88742438269e-05
      42095.71c -2.39699501784e-05
      61147.71c -1.10830735622e-05
      40090.71c -0.90535342614
      45103.71c -1.3813468741e-05
      55133.71c -4.22936858025e-05
      93237.71c -8.59758136013e-07
      55135.71c -3.2306190757e-05
      94240.71c -5.69132116442e-06
      94241.71c -4.96655030195e-07
      94242.71c -2.49160474075e-08
m4012 62152.71c -4.43488035962e-06
      62151.71c -1.72616473916e-06
      62150.71c -9.04581068733e-06
      64155.71c -7.10221118062e-09
      63153.71c -1.67800741988e-06
      60143.71c -4.73471062717e-05
      60145.71c -3.37066417181e-05
      95241.71c -6.56252163243e-08
      95243.71c -1.18059683514e-09
      94239.71c -0.00012847151388
      94238.71c -4.67288383983e-08
      63154.71c -6.31375766069e-08
      44103.71c -1.46788619661e-06
      92238.71c -0.0625776613977
      54135.71c -2.79501751006e-08
      62149.71c -6.19682298263e-07
      54131.71c -2.18938152812e-05
      1001.71c -0.0171203602496
      62147.71c -4.97047222497e-06
      92234.71c -4.03590832029e-08
      92235.71c -0.0140371282137
      92236.71c -0.000264518104371
      43099.71c -3.56930548967e-05
      42095.71c -2.96437161696e-05
      61147.71c -1.35915788625e-05
      40090.71c -0.905546270589
      45103.71c -1.70740944712e-05
      55133.71c -5.22655684593e-05
      93237.71c -1.48280452409e-06
      55135.71c -3.78892026805e-05
      94240.71c -9.60304187674e-06
      94241.71c -1.13218085665e-06
      94242.71c -7.21385154216e-08
m4013 62152.71c -5.42575894162e-06
      62151.71c -1.8459404652e-06
      62150.71c -1.08390118423e-05
      64155.71c -6.95439224933e-09
      63153.71c -2.04431186962e-06
      60143.71c -5.55815861059e-05
      60145.71c -3.98066000346e-05
      95241.71c -1.00084263184e-07
      95243.71c -2.61665754791e-09
      94239.71c -0.000148050371078
      94238.71c -7.59140721651e-08
      63154.71c -8.77883901011e-08
      44103.71c -1.73800589464e-06
      92238.71c -0.0625582844958
      54135.71c -3.11584245209e-08
      62149.71c -6.12062366101e-07
      54131.71c -2.58253555062e-05
      1001.71c -0.0171238529629
      62147.71c -5.82579755503e-06
      92234.71c -4.71836599001e-08
      92235.71c -0.0137424884707
      92236.71c -0.000311987222068
      43099.71c -4.21926653184e-05
      42095.71c -3.50691632023e-05
      61147.71c -1.59761011689e-05
      40090.71c -0.905731010485
      45103.71c -2.01526538242e-05
      55133.71c -6.17663109309e-05
      93237.71c -2.01737280762e-06
      55135.71c -4.24787051106e-05
      94240.71c -1.28976467252e-05
      94241.71c -1.74509161285e-06
      94242.71c -1.34151830316e-07
m4014 62152.71c -5.76799990426e-06
      62151.71c -1.87976617612e-06
      62150.71c -1.14512293452e-05
      64155.71c -6.89359124023e-09
      63153.71c -2.1734908723e-06
      60143.71c -5.83507260219e-05
      60145.71c -4.18781063537e-05
      95241.71c -1.13092046013e-07
      95243.71c -3.32037026553e-09
      94239.71c -0.000154264416958
      94238.71c -8.8004516656e-08
      63154.71c -9.70330405877e-08
      44103.71c -1.83118978454e-06
      92238.71c -0.062551817105
      54135.71c -3.21699645666e-08
      62149.71c -6.0933231496e-07
      54131.71c -2.71583238635e-05
      1001.71c -0.017125045601
      62147.71c -6.11327571481e-06
      92234.71c -4.94114402744e-08
      92235.71c -0.0136424838601
      92236.71c -0.000328039399916
      43099.71c -4.4402176806e-05
      42095.71c -3.69131442356e-05
      61147.71c -1.67801642081e-05
      40090.71c -0.905794092623
      45103.71c -2.11955093859e-05
      55133.71c -6.49955453963e-05
      93237.71c -2.21859069273e-06
      55135.71c -4.39173585109e-05
      94240.71c -1.40895699891e-05
      94241.71c -1.9802188686e-06
      94242.71c -1.61350753689e-07
c **  D-2  Fuel ID= 3330
m4021 62152.71c -2.83405174938e-06
      62151.71c -1.5793589758e-06
      62150.71c -6.2444646515e-06
      64155.71c -7.06803694327e-09
      63153.71c -1.08537659295e-06
      60143.71c -3.42890389424e-05
      60145.71c -2.41094353515e-05
      95241.71c -3.1902354008e-09
      95243.71c -7.85085021456e-12
      94239.71c -8.68740136089e-05
      94238.71c -5.53806033488e-09
      63154.71c -2.29549700898e-08
      44103.71c -1.18016804278e-06
      92238.71c -0.0626219351884
      54135.71c -2.40267345455e-08
      62149.71c -6.22748473417e-07
      54131.71c -1.56737943697e-05
      1001.71c -0.0171149046039
      62147.71c -2.68051277748e-06
      92234.71c -2.62972232661e-08
      92235.71c -0.0145020512368
      92236.71c -0.000187373409176
      43099.71c -2.54439350263e-05
      42095.71c -2.04363479768e-05
      61147.71c -1.07483031344e-05
      40090.71c -0.905257705422
      45103.71c -1.20822757446e-05
      55133.71c -3.72497938201e-05
      93237.71c -5.25930480593e-07
      55135.71c -2.86545302036e-05
      94240.71c -3.47818097169e-06
      94241.71c -1.46485222094e-07
      94242.71c -2.3108674183e-09
m4022 62152.71c -3.75040838656e-06
      62151.71c -1.77142044709e-06
      62150.71c -7.98345481409e-06
      64155.71c -7.01394294833e-09
      63153.71c -1.39586959956e-06
      60143.71c -4.26913547029e-05
      60145.71c -3.01227555622e-05
      95241.71c -8.13753103246e-09
      95243.71c -3.75661289162e-11
      94239.71c -0.000122025075937
      94238.71c -1.26311576114e-08
      63154.71c -3.78977773512e-08
      44103.71c -1.47959486803e-06
      92238.71c -0.0625895422781
      54135.71c -2.82285818682e-08
      62149.71c -6.26171766135e-07
      54131.71c -1.9575398953e-05
      1001.71c -0.0171182881014
      62147.71c -3.3389292173e-06
      92234.71c -3.67632339433e-08
      92235.71c -0.0142092833927
      92236.71c -0.000237405095632
      43099.71c -3.18188756113e-05
      42095.71c -2.55917292946e-05
      61147.71c -1.33441244587e-05
      40090.71c -0.905436668572
      45103.71c -1.51229517132e-05
      55133.71c -4.65752186207e-05
      93237.71c -9.16547644947e-07
      55135.71c -3.39508548866e-05
      94240.71c -6.22137721161e-06
      94241.71c -3.72216328875e-07
      94242.71c -7.51999058424e-09
m4023 62152.71c -4.66779839778e-06
      62151.71c -1.8977657652e-06
      62150.71c -9.63784300485e-06
      64155.71c -6.80465534331e-09
      63153.71c -1.69416369161e-06
      60143.71c -5.05542787735e-05
      60145.71c -3.57985628302e-05
      95241.71c -1.35658774596e-08
      95243.71c -9.07869919736e-11
      94239.71c -0.000142371590185
      94238.71c -2.04770262985e-08
      63154.71c -5.35420048324e-08
      44103.71c -1.75773884117e-06
      92238.71c -0.0625724614418
      54135.71c -3.15643085261e-08
      62149.71c -6.1963404829e-07
      54131.71c -2.3251237322e-05
      1001.71c -0.0171214970019
      62147.71c -3.96233663557e-06
      92234.71c -4.30338127626e-08
      92235.71c -0.0139350351265
      92236.71c -0.000281883896594
      43099.71c -3.78408209311e-05
      42095.71c -3.04799328009e-05
      61147.71c -1.57847747064e-05
      40090.71c -0.905606396769
      45103.71c -1.7981580861e-05
      55133.71c -5.53819491246e-05
      93237.71c -1.25479299205e-06
      55135.71c -3.82966383446e-05
      94240.71c -8.68840041892e-06
      94241.71c -6.19721620681e-07
      94242.71c -1.51242792584e-08
m4024 62152.71c -4.9910699078e-06
      62151.71c -1.93297683925e-06
      62150.71c -1.02084591091e-05
      64155.71c -6.72565204991e-09
      63153.71c -1.7993225407e-06
      60143.71c -5.32428793293e-05
      60145.71c -3.77482083083e-05
      95241.71c -1.5803843835e-08
      95243.71c -1.18500884671e-10
      94239.71c -0.000148942269641
      94238.71c -2.37331747207e-08
      63154.71c -5.95085721577e-08
      44103.71c -1.85138793133e-06
      92238.71c -0.0625668436942
      54135.71c -3.25809234268e-08
      62149.71c -6.17294461854e-07
      54131.71c -2.45135854242e-05
      1001.71c -0.0171226028279
      62147.71c -4.17724194546e-06
      92234.71c -4.52803279122e-08
      92235.71c -0.0138408925051
      92236.71c -0.000297088164317
      43099.71c -3.99115105415e-05
      42095.71c -3.21698912738e-05
      61147.71c -1.66172018759e-05
      40090.71c -0.905664887159
      45103.71c -1.8965386346e-05
      55133.71c -5.84098803164e-05
      93237.71c -1.38149847346e-06
      55135.71c -3.96794758774e-05
      94240.71c -9.60260473809e-06
      94241.71c -7.21085114723e-07
      94242.71c -1.86680324765e-08
c **  D-3  Fuel ID= 10706
m4031 92238.71c -0.062672852401
      1001.71c -0.0171016621372
      92235.71c -0.0156682131003
      40090.71c -0.904557272362
m4032 92238.71c -0.062672852401
      1001.71c -0.0171016621372
      92235.71c -0.0156682131003
      40090.71c -0.904557272362
m4033 92238.71c -0.062672852401
      1001.71c -0.0171016621372
      92235.71c -0.0156682131003
      40090.71c -0.904557272362
m4034 92238.71c -0.062672852401
      1001.71c -0.0171016621372
      92235.71c -0.0156682131003
      40090.71c -0.904557272362
c **  D-5  Fuel ID= 3144
m4051 62152.71c -3.55781796204e-06
      62151.71c -1.55855648463e-06
      62150.71c -7.37480861139e-06
      64155.71c -7.2195476937e-09
      63153.71c -1.33743691631e-06
      60143.71c -3.95255555082e-05
      60145.71c -2.79840075005e-05
      95241.71c -3.22393668003e-08
      95243.71c -3.13521703087e-10
      94239.71c -9.48309237354e-05
      94238.71c -2.22634012678e-08
      63154.71c -4.14289421314e-08
      44103.71c -1.178954574e-06
      92238.71c -0.0626127893135
      54135.71c -2.39023934924e-08
      62149.71c -6.15799977557e-07
      54131.71c -1.82014745015e-05
      1001.71c -0.0171170951552
      62147.71c -4.29615032045e-06
      92234.71c -2.98920356454e-08
      92235.71c -0.0143154683288
      92236.71c -0.000216719436873
      43099.71c -2.96054799166e-05
      42095.71c -2.47218616971e-05
      61147.71c -1.1234536448e-05
      40090.71c -0.905373570133
      45103.71c -1.41865906543e-05
      55133.71c -4.33701586799e-05
      93237.71c -9.10428630423e-07
      55135.71c -3.3105143547e-05
      94240.71c -6.03469776438e-06
      94241.71c -5.42361736252e-07
      94242.71c -2.7628601367e-08
m4052 62152.71c -4.57796930288e-06
      62151.71c -1.73216028803e-06
      62150.71c -9.28748594348e-06
      64155.71c -7.1883177876e-09
      63153.71c -1.73153103367e-06
      60143.71c -4.84742034532e-05
      60145.71c -3.45299990976e-05
      95241.71c -7.26321538498e-08
      95243.71c -1.31437505271e-09
      94239.71c -0.000130762536731
      94238.71c -5.11139051501e-08
      63154.71c -6.70245095072e-08
      44103.71c -1.467097674e-06
      92238.71c -0.0625750290104
      54135.71c -2.79081375798e-08
      62149.71c -6.18392952205e-07
      54131.71c -2.2431754104e-05
      1001.71c -0.0171208253044
      62147.71c -5.2412067563e-06
      92234.71c -4.12954349022e-08
      92235.71c -0.0139974521217
      92236.71c -0.000270870893546
      43099.71c -3.65769775056e-05
      42095.71c -3.05564587322e-05
      61147.71c -1.37662622208e-05
      40090.71c -0.905570868708
      45103.71c -1.75244678036e-05
      55133.71c -5.35658452586e-05
      93237.71c -1.57059457159e-06
      55135.71c -3.88040460088e-05
      94240.71c -1.01529437872e-05
      94241.71c -1.23360745414e-06
      94242.71c -7.99444074903e-08
m4053 62152.71c -5.59232586826e-06
      62151.71c -1.84994869141e-06
      62150.71c -1.11192365456e-05
      64155.71c -7.03663855924e-09
      63153.71c -2.11063558688e-06
      60143.71c -5.68547670347e-05
      60145.71c -4.0754223985e-05
      95241.71c -1.10689406045e-07
      95243.71c -2.91273513947e-09
      94239.71c -0.000150462781816
      94238.71c -8.30476543498e-08
      63154.71c -9.31149994281e-08
      44103.71c -1.73636882001e-06
      92238.71c -0.0625551169095
      54135.71c -3.10957070447e-08
      62149.71c -6.1057141058e-07
      54131.71c -2.64420423383e-05
      1001.71c -0.0171243939576
      62147.71c -6.13835582042e-06
      92234.71c -4.82531471901e-08
      92235.71c -0.0136968366385
      92236.71c -0.000319270581535
      43099.71c -4.32126833766e-05
      42095.71c -3.61273330152e-05
      61147.71c -1.61692024294e-05
      40090.71c -0.905759625284
      45103.71c -2.06698637164e-05
      55133.71c -6.3265759197e-05
      93237.71c -2.13633595682e-06
      55135.71c -4.34776803128e-05
      94240.71c -1.36040755066e-05
      94241.71c -1.89769613118e-06
      94242.71c -1.48590831274e-07
m4054 62152.71c -5.9421252256e-06
      62151.71c -1.8831588631e-06
      62150.71c -1.17443379111e-05
      64155.71c -6.97553117573e-09
      63153.71c -2.24440508277e-06
      60143.71c -5.96700936028e-05
      60145.71c -4.28666854149e-05
      95241.71c -1.25040360706e-07
      95243.71c -3.695870711e-09
      94239.71c -0.000156700439936
      94238.71c -9.62764090435e-08
      63154.71c -1.02893033037e-07
      44103.71c -1.82920267617e-06
      92238.71c -0.062548469374
      54135.71c -3.20995171941e-08
      62149.71c -6.07772795496e-07
      54131.71c -2.78008399878e-05
      1001.71c -0.0171256120657
      62147.71c -6.43964779866e-06
      92234.71c -5.05212677051e-08
      92235.71c -0.0135948609302
      92236.71c -0.000335624717416
      43099.71c -4.54672111511e-05
      42095.71c -3.80198818774e-05
      61147.71c -1.69788323929e-05
      40090.71c -0.905824054608
      45103.71c -2.17345988412e-05
      55133.71c -6.65609293164e-05
      93237.71c -2.34935138645e-06
      55135.71c -4.49410999971e-05
      94240.71c -1.48496956222e-05
      94241.71c -2.15181053397e-06
      94242.71c -1.78681961245e-07
c **  D-6  Fuel ID= 6224
m4061 62152.71c -1.02018233071e-06
      62151.71c -8.19601615592e-07
      62150.71c -2.38532451006e-06
      64155.71c -2.9066256882e-09
      63153.71c -4.3907987815e-07
      60143.71c -1.40825355147e-05
      60145.71c -1.00735383359e-05
      95241.71c -2.4868872718e-10
      95243.71c -3.08492297324e-13
      94239.71c -3.76205779212e-05
      94238.71c -8.91774538364e-10
      63154.71c -5.54787733837e-09
      44103.71c -8.34310309408e-07
      92238.71c -0.0626520406177
      54135.71c -1.73386734692e-08
      62149.71c -4.67964658172e-07
      54131.71c -6.47141204914e-06
      1001.71c -0.0171072217086
      62147.71c -6.67133503977e-07
      92234.71c -1.11700098198e-08
      92235.71c -0.0151804940673
      92236.71c -7.85061323285e-05
      43099.71c -1.05738336535e-05
      42095.71c -6.90150417385e-06
      61147.71c -4.87937443594e-06
      40090.71c -0.904851334465
      45103.71c -4.71006490434e-06
      55133.71c -1.54042977224e-05
      93237.71c -1.47338319773e-07
      55135.71c -1.2009268559e-05
      94240.71c -8.37363620283e-07
      94241.71c -2.00299489118e-08
      94242.71c -1.6901223554e-10
m4062 62152.71c -1.36306546259e-06
      62151.71c -9.65042632956e-07
      62150.71c -3.13965174505e-06
      64155.71c -3.10545284657e-09
      63153.71c -5.63087905799e-07
      60143.71c -1.77586304292e-05
      60145.71c -1.27272939941e-05
      95241.71c -6.61464555917e-10
      95243.71c -1.55744470031e-12
      94239.71c -5.40434436049e-05
      94238.71c -1.98809366942e-09
      63154.71c -9.33761948632e-09
      44103.71c -1.05688027919e-06
      92238.71c -0.0626384068833
      54135.71c -2.06550326073e-08
      62149.71c -4.73242038315e-07
      54131.71c -8.17567201723e-06
      1001.71c -0.0171087128823
      62147.71c -8.41360634756e-07
      92234.71c -1.5606476183e-08
      92235.71c -0.0150509600623
      92236.71c -0.000100664957078
      43099.71c -1.33670739917e-05
      42095.71c -8.73446066573e-06
      61147.71c -6.14242753621e-06
      40090.71c -0.904930207032
      45103.71c -5.95982677107e-06
      55133.71c -1.94716749517e-05
      93237.71c -2.5143365384e-07
      55135.71c -1.43608969665e-05
      94240.71c -1.54785890634e-06
      94241.71c -5.32269865184e-08
      94242.71c -5.76215277541e-10
m4063 62152.71c -1.70789944796e-06
      62151.71c -1.07750538611e-06
      62150.71c -3.85482050252e-06
      64155.71c -3.18598937455e-09
      63153.71c -6.80265345712e-07
      60143.71c -2.11902151735e-05
      60145.71c -1.52148920671e-05
      95241.71c -1.12574058601e-09
      95243.71c -3.86056114804e-12
      94239.71c -6.39645686158e-05
      94238.71c -3.18478383899e-09
      63154.71c -1.33381602344e-08
      44103.71c -1.26358496618e-06
      92238.71c -0.0626314426069
      54135.71c -2.33228649833e-08
      62149.71c -4.70808708968e-07
      54131.71c -9.77129051108e-06
      1001.71c -0.0171101139794
      62147.71c -1.0049803669e-06
      92234.71c -1.85278185953e-08
      92235.71c -0.0149304795375
      92236.71c -0.00012028489366
      43099.71c -1.59864348246e-05
      42095.71c -1.04593068764e-05
      61147.71c -7.32344481848e-06
      40090.71c -0.905004315184
      45103.71c -7.12896322964e-06
      55133.71c -2.32855507874e-05
      93237.71c -3.41356004416e-07
      55135.71c -1.62858668948e-05
      94240.71c -2.19764539836e-06
      94241.71c -9.05271094906e-08
      94242.71c -1.18218261274e-09
m4064 62152.71c -1.83191156194e-06
      62151.71c -1.1125656463e-06
      62150.71c -4.10415932138e-06
      64155.71c -3.20093678969e-09
      63153.71c -7.21651448964e-07
      60143.71c -2.23802069353e-05
      60145.71c -1.60797171171e-05
      95241.71c -1.32415458918e-09
      95243.71c -5.08313837132e-12
      94239.71c -6.72292598361e-05
      94238.71c -3.68911739634e-09
      63154.71c -1.48859606452e-08
      44103.71c -1.3353952123e-06
      92238.71c -0.0626289928119
      54135.71c -2.41753769304e-08
      62149.71c -4.69758155371e-07
      54131.71c -1.03257581064e-05
      1001.71c -0.0171106054969
      62147.71c -1.06183817233e-06
      92234.71c -1.96120743219e-08
      92235.71c -0.0148886019987
      92236.71c -0.000127093672588
      43099.71c -1.68975614327e-05
      42095.71c -1.10604401817e-05
      61147.71c -7.73296313931e-06
      40090.71c -0.905030312994
      45103.71c -7.5356953703e-06
      55133.71c -2.46121423068e-05
      93237.71c -3.75336921145e-07
      55135.71c -1.69069764592e-05
      94240.71c -2.44486423342e-06
      94241.71c -1.06457197454e-07
      94242.71c -1.47412907255e-09
c **  D-7  Fuel ID= 3018
m4071 62152.71c -3.46977000019e-06
      62151.71c -1.70472590503e-06
      62150.71c -7.4424602293e-06
      64155.71c -8.04368481609e-09
      63153.71c -1.28954372484e-06
      60143.71c -4.0267081913e-05
      60145.71c -2.82535822396e-05
      95241.71c -5.90859465403e-09
      95243.71c -1.73264758672e-11
      94239.71c -0.000100288598179
      94238.71c -8.58586963494e-09
      63154.71c -3.12805681547e-08
      44103.71c -1.18601174947e-06
      92238.71c -0.062612601044
      54135.71c -2.39826987979e-08
      62149.71c -6.16750235439e-07
      54131.71c -1.84088946274e-05
      1001.71c -0.0171171773996
      62147.71c -3.60473393842e-06
      92234.71c -3.07047895805e-08
      92235.71c -0.0143019909552
      92236.71c -0.000219393018077
      43099.71c -2.98569488359e-05
      42095.71c -2.49294575487e-05
      61147.71c -1.2153679444e-05
      40090.71c -0.905377920282
      45103.71c -1.43522241009e-05
      55133.71c -4.37527042296e-05
      93237.71c -6.98369256157e-07
      55135.71c -3.35566203681e-05
      94240.71c -4.73949735326e-06
      94241.71c -2.3277408389e-07
      94242.71c -4.34964804839e-09
m4072 62152.71c -4.57789105739e-06
      62151.71c -1.88404943668e-06
      62150.71c -9.47364596043e-06
      64155.71c -7.83468775044e-09
      63153.71c -1.66540308105e-06
      60143.71c -5.00172509362e-05
      60145.71c -3.52425547241e-05
      95241.71c -1.49337451888e-08
      95243.71c -8.22641300287e-11
      94239.71c -0.000139994972356
      94238.71c -1.97277510546e-08
      63154.71c -5.15422216777e-08
      44103.71c -1.48463665091e-06
      92238.71c -0.0625745184796
      54135.71c -2.80877905989e-08
      62149.71c -6.18857128466e-07
      54131.71c -2.29498232137e-05
      1001.71c -0.0171211174565
      62147.71c -4.48057424933e-06
      92234.71c -4.27575093824e-08
      92235.71c -0.0139620962166
      92236.71c -0.000277398277428
      43099.71c -3.72814925626e-05
      42095.71c -3.11740373165e-05
      61147.71c -1.5047249915e-05
      40090.71c -0.905586321492
      45103.71c -1.79362102275e-05
      55133.71c -5.46218878429e-05
      93237.71c -1.22285400026e-06
      55135.71c -3.9695824897e-05
      94240.71c -8.41394957954e-06
      94241.71c -5.8590500371e-07
      94242.71c -1.40416302523e-08
m4073 62152.71c -5.67631544276e-06
      62151.71c -1.99225475912e-06
      62150.71c -1.13971240216e-05
      64155.71c -7.50713121704e-09
      63153.71c -2.02622285276e-06
      60143.71c -5.90876866279e-05
      60145.71c -4.18120120788e-05
      95241.71c -2.47230426488e-08
      95243.71c -1.97321125376e-10
      94239.71c -0.000162474655411
      94238.71c -3.20833610904e-08
      63154.71c -7.26402037246e-08
      44103.71c -1.75873407221e-06
      92238.71c -0.0625544094612
      54135.71c -3.1283744639e-08
      62149.71c -6.11056302274e-07
      54131.71c -2.72099923484e-05
      1001.71c -0.0171248407635
      62147.71c -5.30761137441e-06
      92234.71c -4.99911519137e-08
      92235.71c -0.0136448981887
      92236.71c -0.000328738594372
      43099.71c -4.42668525053e-05
      42095.71c -3.70756929538e-05
      61147.71c -1.77537707528e-05
      40090.71c -0.905783258168
      45103.71c -2.12896901419e-05
      55133.71c -6.48458561694e-05
      93237.71c -1.6766799179e-06
      55135.71c -4.47002762318e-05
      94240.71c -1.16784999355e-05
      94241.71c -9.67351744921e-07
      94242.71c -2.80628604865e-08
m4074 62152.71c -6.06112148424e-06
      62151.71c -2.02098590234e-06
      62150.71c -1.20603646153e-05
      64155.71c -7.38933395338e-09
      63153.71c -2.15415337602e-06
      60143.71c -6.21802677454e-05
      60145.71c -4.40664166456e-05
      95241.71c -2.87270649354e-08
      95243.71c -2.56733765576e-10
      94239.71c -0.000169734603957
      94238.71c -3.72378867973e-08
      63154.71c -8.06900422607e-08
      44103.71c -1.85255259062e-06
      92238.71c -0.0625475874434
      54135.71c -3.22745711311e-08
      62149.71c -6.08240419836e-07
      54131.71c -2.86702639503e-05
      1001.71c -0.0171261262575
      62147.71c -5.59155064771e-06
      92234.71c -5.24955399311e-08
      92235.71c -0.0135360932813
      92236.71c -0.000346275409442
      43099.71c -4.66666440054e-05
      42095.71c -3.91086448414e-05
      61147.71c -1.86746864157e-05
      40090.71c -0.905851251727
      45103.71c -2.24405390562e-05
      55133.71c -6.83567669826e-05
      93237.71c -1.84828308657e-06
      55135.71c -4.62891807679e-05
      94240.71c -1.28839418887e-05
      94241.71c -1.12303133938e-06
      94242.71c -3.45710210262e-08
c **  D-8  Fuel ID= 3105
m4081 62152.71c -3.40428024027e-06
      62151.71c -1.54806300477e-06
      62150.71c -7.11350113792e-06
      64155.71c -7.09564538654e-09
      63153.71c -1.28514165532e-06
      60143.71c -3.82728908246e-05
      60145.71c -2.70845718107e-05
      95241.71c -2.81143025519e-08
      95243.71c -2.71455670626e-10
      94239.71c -9.24251969028e-05
      94238.71c -1.97639801045e-08
      63154.71c -3.82428434815e-08
      44103.71c -1.17931796619e-06
      92238.71c -0.0626149471835
      54135.71c -2.3934360462e-08
      62149.71c -6.17228812045e-07
      54131.71c -1.76115015932e-05
      1001.71c -0.0171165938531
      62147.71c -3.99948024896e-06
      92234.71c -2.89709495788e-08
      92235.71c -0.0143588421711
      92236.71c -0.00020982932819
      43099.71c -2.8642960818e-05
      42095.71c -2.37321277283e-05
      61147.71c -1.10351672943e-05
      40090.71c -0.905347054787
      45103.71c -1.36954538345e-05
      55133.71c -4.19532076867e-05
      93237.71c -8.43731541357e-07
      55135.71c -3.2053489601e-05
      94240.71c -5.58271441713e-06
      94241.71c -4.82198434507e-07
      94242.71c -2.4058092432e-08
m4082 62152.71c -4.38962269004e-06
      62151.71c -1.72426840418e-06
      62150.71c -8.96937110619e-06
      64155.71c -7.07497648115e-09
      63153.71c -1.66107841075e-06
      60143.71c -4.69906161708e-05
      60145.71c -3.34462216229e-05
      95241.71c -6.34089887386e-08
      95243.71c -1.13828405939e-09
      94239.71c -0.000127746885178
      94238.71c -4.53418835603e-08
      63154.71c -6.19081767605e-08
      44103.71c -1.4681355988e-06
      92238.71c -0.0625784939967
      54135.71c -2.79634711679e-08
      62149.71c -6.20090106138e-07
      54131.71c -2.17236703564e-05
      1001.71c -0.0171202131571
      62147.71c -4.88484147194e-06
      92234.71c -4.00629240809e-08
      92235.71c -0.0140496773836
      92236.71c -0.000262508777677
      43099.71c -3.54134785965e-05
      42095.71c -2.93550243925e-05
      61147.71c -1.35363281804e-05
      40090.71c -0.905538490438
      45103.71c -1.69316456767e-05
      55133.71c -5.18543032951e-05
      93237.71c -1.45503736842e-06
      55135.71c -3.75998464512e-05
      94240.71c -9.42911314126e-06
      94241.71c -1.10010059014e-06
      94242.71c -6.96695861253e-08
m4083 62152.71c -5.37307540957e-06
      62151.71c -1.84467270144e-06
      62150.71c -1.07503794382e-05
      64155.71c -6.92837852505e-09
      63153.71c -2.0233343097e-06
      60143.71c -5.51788911074e-05
      60145.71c -3.95068756069e-05
      95241.71c -9.67299575306e-08
      95243.71c -2.52301102637e-09
      94239.71c -0.000147287348544
      94238.71c -7.36577880801e-08
      63154.71c -8.61036343163e-08
      44103.71c -1.73852368576e-06
      92238.71c -0.0625592863731
      54135.71c -3.11782614612e-08
      62149.71c -6.12533941128e-07
      54131.71c -2.56303033362e-05
      1001.71c -0.0171236818515
      62147.71c -5.72693835336e-06
      92234.71c -4.68453912591e-08
      92235.71c -0.0137569277102
      92236.71c -0.000309683564855
      43099.71c -4.1870043324e-05
      42095.71c -3.47344741714e-05
      61147.71c -1.59150250798e-05
      40090.71c -0.905721959896
      45103.71c -1.99890652627e-05
      55133.71c -6.12920497292e-05
      93237.71c -1.97974589687e-06
      55135.71c -4.21627387711e-05
      94240.71c -1.26742100316e-05
      94241.71c -1.69682425753e-06
      94242.71c -1.2958491187e-07
m4084 62152.71c -5.71292572398e-06
      62151.71c -1.87869310155e-06
      62150.71c -1.13585218982e-05
      64155.71c -6.86767441926e-09
      63153.71c -2.15106138312e-06
      60143.71c -5.79334226253e-05
      60145.71c -4.15654282147e-05
      95241.71c -1.09312907905e-07
      95243.71c -3.20160305193e-09
      94239.71c -0.000153493926097
      94238.71c -8.53881958983e-08
      63154.71c -9.51795808029e-08
      44103.71c -1.83181828799e-06
      92238.71c -0.0625528759604
      54135.71c -3.21922463994e-08
      62149.71c -6.09825576109e-07
      54131.71c -2.69551021344e-05
      1001.71c -0.0171248664336
      62147.71c -6.01004733396e-06
      92234.71c -4.90604124311e-08
      92235.71c -0.0136575465392
      92236.71c -0.000325640236255
      43099.71c -4.4065316589e-05
      42095.71c -3.65630936631e-05
      61147.71c -1.67173273539e-05
      40090.71c -0.905784615932
      45103.71c -2.1025000527e-05
      55133.71c -6.45004293754e-05
      93237.71c -2.17723233103e-06
      55135.71c -4.35935588318e-05
      94240.71c -1.38491494929e-05
      94241.71c -1.92594605956e-06
      94242.71c -1.5586905785e-07
c **  D-9  Fuel ID= 2452
m4091 62152.71c -2.9420137195e-06
      62151.71c -1.51646956719e-06
      62150.71c -6.32676485437e-06
      64155.71c -6.72260423557e-09
      63153.71c -1.12769273613e-06
      60143.71c -3.45014075806e-05
      60145.71c -2.43765792593e-05
      95241.71c -1.56946894693e-08
      95243.71c -1.44804589425e-10
      94239.71c -8.51821105787e-05
      94238.71c -1.22388018506e-08
      63154.71c -2.86502383503e-08
      44103.71c -1.18041205593e-06
      92238.71c -0.0626214440304
      54135.71c -2.40306056037e-08
      62149.71c -6.21530701768e-07
      54131.71c -1.58352298048e-05
      1001.71c -0.0171150845485
      62147.71c -3.10627537282e-06
      92234.71c -2.61977727569e-08
      92235.71c -0.0144894307649
      92236.71c -0.000189084806703
      43099.71c -2.57450387328e-05
      42095.71c -2.07522679718e-05
      61147.71c -1.04349129467e-05
      40090.71c -0.905267223222
      45103.71c -1.22167546178e-05
      55133.71c -3.76870964135e-05
      93237.71c -6.42922053299e-07
      55135.71c -2.88872031337e-05
      94240.71c -4.22189723839e-06
      94241.71c -3.01060666841e-07
      94242.71c -1.33081169627e-08
m4092 62152.71c -3.82255466005e-06
      62151.71c -1.70050776967e-06
      62150.71c -8.01160100686e-06
      64155.71c -6.73373205741e-09
      63153.71c -1.44896188006e-06
      60143.71c -4.25238784638e-05
      60145.71c -3.01832186507e-05
      95241.71c -3.56401746837e-08
      95243.71c -6.0811486137e-10
      94239.71c -0.00011866745678
      94238.71c -2.79636633599e-08
      63154.71c -4.65040838085e-08
      44103.71c -1.47126054998e-06
      92238.71c -0.0625889262682
      54135.71c -2.81300677866e-08
      62149.71c -6.25199843569e-07
      54131.71c -1.95917941855e-05
      1001.71c -0.0171183701236
      62147.71c -3.8119081353e-06
      92234.71c -3.63521194489e-08
      92235.71c -0.0142069155519
      92236.71c -0.000237332383469
      43099.71c -3.19104528487e-05
      42095.71c -2.57377838492e-05
      61147.71c -1.28440500377e-05
      40090.71c -0.905441006964
      45103.71c -1.51467955508e-05
      55133.71c -4.6701246838e-05
      93237.71c -1.10712139284e-06
      55135.71c -3.39742804772e-05
      94240.71c -7.24982670633e-06
      94241.71c -6.98142342945e-07
      94242.71c -3.87344786551e-08
m4093 62152.71c -4.71296305709e-06
      62151.71c -1.82878791755e-06
      62150.71c -9.6398361155e-06
      64155.71c -6.60243263526e-09
      63153.71c -1.76049038317e-06
      60143.71c -5.01332168249e-05
      60145.71c -3.57513985277e-05
      95241.71c -5.47012910772e-08
      95243.71c -1.34964198238e-09
      94239.71c -0.000137726854397
      94238.71c -4.53870754453e-08
      63154.71c -6.49940378046e-08
      44103.71c -1.74501148745e-06
      92238.71c -0.0625718396626
      54135.71c -3.14268136893e-08
      62149.71c -6.18442666084e-07
      54131.71c -2.31863452005e-05
      1001.71c -0.017121537865
      62147.71c -4.48825564478e-06
      92234.71c -4.26069641886e-08
      92235.71c -0.01393784801
      92236.71c -0.000280819277995
      43099.71c -3.78276650825e-05
      42095.71c -3.05408987804e-05
      61147.71c -1.51497559477e-05
      40090.71c -0.905608558135
      45103.71c -1.79393387921e-05
      55133.71c -5.53496676351e-05
      93237.71c -1.5082894928e-06
      55135.71c -3.8203754331e-05
      94240.71c -9.87460044429e-06
      94241.71c -1.09204556811e-06
      94242.71c -7.2362490337e-08
m4094 62152.71c -5.02285910746e-06
      62151.71c -1.86524772778e-06
      62150.71c -1.01969192387e-05
      64155.71c -6.54294270529e-09
      63153.71c -1.87002512195e-06
      60143.71c -5.2704708527e-05
      60145.71c -3.76476441574e-05
      95241.71c -6.19611900155e-08
      95243.71c -1.7134776034e-09
      94239.71c -0.000143839855561
      94238.71c -5.26063078401e-08
      63154.71c -7.195616257e-08
      44103.71c -1.83969328941e-06
      92238.71c -0.0625661431715
      54135.71c -3.24714325604e-08
      62149.71c -6.1600602311e-07
      54131.71c -2.44087813319e-05
      1001.71c -0.0171226215081
      62147.71c -4.71661983069e-06
      92234.71c -4.46621155366e-08
      92235.71c -0.0138462783897
      92236.71c -0.000295579275896
      43099.71c -3.9844536743e-05
      42095.71c -3.21770417436e-05
      61147.71c -1.59299962463e-05
      40090.71c -0.905665875213
      45103.71c -1.88885643466e-05
      55133.71c -5.82967412673e-05
      93237.71c -1.65902171813e-06
      55135.71c -3.9536424475e-05
      94240.71c -1.08367368252e-05
      94241.71c -1.24592043757e-06
      94242.71c -8.71846892189e-08
c **  D-10  Fuel ID= 10893
m4101 92238.71c -0.0626727222977
      1001.71c -0.0171016644307
      92235.71c -0.0156682195967
      40090.71c -0.904557393675
m4102 92238.71c -0.0626727222977
      1001.71c -0.0171016644307
      92235.71c -0.0156682195967
      40090.71c -0.904557393675
m4103 92238.71c -0.0626727222977
      1001.71c -0.0171016644307
      92235.71c -0.0156682195967
      40090.71c -0.904557393675
m4104 92238.71c -0.0626727222977
      1001.71c -0.0171016644307
      92235.71c -0.0156682195967
      40090.71c -0.904557393675
c **  D-11  Fuel ID= 2435
m4111 62152.71c -2.1986381629e-06
      62151.71c -1.35525854351e-06
      62150.71c -4.89628210329e-06
      64155.71c -5.54038194914e-09
      63153.71c -8.68237849636e-07
      60143.71c -2.74570314706e-05
      60145.71c -1.9429004024e-05
      95241.71c -2.41972010769e-09
      95243.71c -6.94929375033e-12
      94239.71c -6.96592254906e-05
      94238.71c -3.93592497762e-09
      63154.71c -1.65664031978e-08
      44103.71c -1.16853314028e-06
      92238.71c -0.0626330711893
      54135.71c -2.3936618141e-08
      62149.71c -6.28155140473e-07
      54131.71c -1.25803534601e-05
      1001.71c -0.0171123565562
      62147.71c -1.92169658671e-06
      92234.71c -2.09699550002e-08
      92235.71c -0.0147283469183
      92236.71c -0.000150876893193
      43099.71c -2.04700677598e-05
      42095.71c -1.54146899266e-05
      61147.71c -8.85471235646e-06
      40090.71c -0.905122931678
      45103.71c -9.51935108851e-06
      55133.71c -2.99193583046e-05
      93237.71c -3.85008563115e-07
      55135.71c -2.30305515978e-05
      94240.71c -2.48242914533e-06
      94241.71c -1.03023030795e-07
      94242.71c -1.78164759051e-09
m4112 62152.71c -2.86123546873e-06
      62151.71c -1.53534496494e-06
      62150.71c -6.21187915074e-06
      64155.71c -5.63510104498e-09
      63153.71c -1.0995288399e-06
      60143.71c -3.38221029787e-05
      60145.71c -2.40039112386e-05
      95241.71c -5.95885502347e-09
      95243.71c -3.13067836436e-11
      94239.71c -9.78081178604e-05
      94238.71c -8.72949680627e-09
      63154.71c -2.67301898789e-08
      44103.71c -1.44939115814e-06
      92238.71c -0.0626076848715
      54135.71c -2.79962968887e-08
      62149.71c -6.34009703449e-07
      54131.71c -1.55378049751e-05
      1001.71c -0.017114930438
      62147.71c -2.36520748223e-06
      92234.71c -2.90349538719e-08
      92235.71c -0.0145053392597
      92236.71c -0.000189152743515
      43099.71c -2.53101331226e-05
      42095.71c -1.90774445029e-05
      61147.71c -1.08835255551e-05
      40090.71c -0.905259071867
      45103.71c -1.17799990207e-05
      55133.71c -3.69882322914e-05
      93237.71c -6.55885738386e-07
      55135.71c -2.70635298224e-05
      94240.71c -4.36991179958e-06
      94241.71c -2.5397638117e-07
      94242.71c -5.53156968609e-09
m4113 62152.71c -3.5521718183e-06
      62151.71c -1.67116412936e-06
      62150.71c -7.51745496343e-06
      64155.71c -5.58940001152e-09
      63153.71c -1.32930071432e-06
      60143.71c -4.00350756361e-05
      60145.71c -2.85026324356e-05
      95241.71c -9.83369101557e-09
      95243.71c -7.47777341826e-11
      94239.71c -0.000114362566871
      94238.71c -1.40996871489e-08
      63154.71c -3.76533784331e-08
      44103.71c -1.72249747848e-06
      92238.71c -0.0625944656951
      54135.71c -3.14073338334e-08
      62149.71c -6.29040138854e-07
      54131.71c -1.84409949209e-05
      1001.71c -0.0171174739193
      62147.71c -2.80184873598e-06
      92234.71c -3.42480163249e-08
      92235.71c -0.0142878149676
      92236.71c -0.000224449345546
      43099.71c -3.00715276039e-05
      42095.71c -2.26896027491e-05
      61147.71c -1.2874417065e-05
      40090.71c -0.90539360408
      45103.71c -1.39926210539e-05
      55133.71c -4.39409000152e-05
      93237.71c -8.94395409629e-07
      55135.71c -3.05213390049e-05
      94240.71c -6.07904468201e-06
      94241.71c -4.19498767166e-07
      94242.71c -1.09919691064e-08
m4114 62152.71c -3.79394597056e-06
      62151.71c -1.71102493413e-06
      62150.71c -7.96360164819e-06
      64155.71c -5.56030142721e-09
      63153.71c -1.40928897779e-06
      60143.71c -4.21414356393e-05
      60145.71c -3.00343621748e-05
      95241.71c -1.1429295656e-08
      95243.71c -9.70619918311e-11
      94239.71c -0.000119794649667
      94238.71c -1.62943447416e-08
      63154.71c -4.179138469e-08
      44103.71c -1.81500298687e-06
      92238.71c -0.0625900103473
      54135.71c -3.24668299161e-08
      62149.71c -6.27265492146e-07
      54131.71c -1.94288964582e-05
      1001.71c -0.0171183428926
      62147.71c -2.95052671558e-06
      92234.71c -3.59404445972e-08
      92235.71c -0.0142137614664
      92236.71c -0.00023646584516
      43099.71c -3.16942312659e-05
      42095.71c -2.39244869234e-05
      61147.71c -1.35488674316e-05
      40090.71c -0.905439566637
      45103.71c -1.47467008692e-05
      55133.71c -4.63100873832e-05
      93237.71c -9.81060902016e-07
      55135.71c -3.16150579755e-05
      94240.71c -6.71746043229e-06
      94241.71c -4.87743209443e-07
      94242.71c -1.35347746389e-08
c **  D-12  Fuel ID= 3876
m4121 62152.71c -2.96017521683e-06
      62151.71c -1.60423117633e-06
      62150.71c -6.48214125756e-06
      64155.71c -7.26160082566e-09
      63153.71c -1.12588237768e-06
      60143.71c -3.54750541419e-05
      60145.71c -2.49316143191e-05
      95241.71c -3.72954490523e-09
      95243.71c -9.73076912842e-12
      94239.71c -8.95354031762e-05
      94238.71c -6.14273115653e-09
      63154.71c -2.46067290179e-08
      44103.71c -1.1813274063e-06
      92238.71c -0.0626200833387
      54135.71c -2.40179980633e-08
      62149.71c -6.21558451625e-07
      54131.71c -1.62164252096e-05
      1001.71c -0.0171153555157
      62147.71c -2.86387384563e-06
      92234.71c -2.71716633909e-08
      92235.71c -0.0144623602316
      92236.71c -0.000193725946791
      43099.71c -2.63194559082e-05
      42095.71c -2.13277594757e-05
      61147.71c -1.10271230881e-05
      40090.71c -0.905281555476
      45103.71c -1.25326226668e-05
      55133.71c -3.85399402149e-05
      93237.71c -5.60141510824e-07
      55135.71c -2.96270815005e-05
      94240.71c -3.72842012307e-06
      94241.71c -1.63604520523e-07
      94242.71c -2.71535176669e-09
m4122 62152.71c -3.91457699989e-06
      62151.71c -1.79376550118e-06
      62150.71c -8.27910162666e-06
      64155.71c -7.17677480066e-09
      63153.71c -1.44934375614e-06
      60143.71c -4.4144777558e-05
      60145.71c -3.11384992853e-05
      95241.71c -9.48586749102e-09
      95243.71c -4.64339990478e-11
      94239.71c -0.000125590217638
      94238.71c -1.40390878872e-08
      63154.71c -4.06047699972e-08
      44103.71c -1.48059513369e-06
      92238.71c -0.0625865616282
      54135.71c -2.82006495522e-08
      62149.71c -6.24720576925e-07
      54131.71c -2.02448686243e-05
      1001.71c -0.017118849432
      62147.71c -3.56542614398e-06
      92234.71c -3.7952469587e-08
      92235.71c -0.0141602426365
      92236.71c -0.000245339552061
      43099.71c -3.2902632748e-05
      42095.71c -2.66992325686e-05
      61147.71c -1.36820164223e-05
      40090.71c -0.905466358997
      45103.71c -1.56810887781e-05
      55133.71c -4.81716393974e-05
      93237.71c -9.77317364213e-07
      55135.71c -3.50906295229e-05
      94240.71c -6.65637310693e-06
      94241.71c -4.14611142286e-07
      94242.71c -8.81385277454e-09
m4123 62152.71c -4.86788336708e-06
      62151.71c -1.91651193071e-06
      62150.71c -9.98687596367e-06
      64155.71c -6.9440232046e-09
      63153.71c -1.76004264467e-06
      60143.71c -5.22472661722e-05
      60145.71c -3.69916024663e-05
      95241.71c -1.57794057936e-08
      95243.71c -1.11922855683e-10
      94239.71c -0.000146359942398
      94238.71c -2.27796677383e-08
      63154.71c -5.73309963602e-08
      44103.71c -1.75793629026e-06
      92238.71c -0.062568880015
      54135.71c -3.15086459897e-08
      62149.71c -6.17932264412e-07
      54131.71c -2.40366354289e-05
      1001.71c -0.0171221603883
      62147.71c -4.2292327244e-06
      92234.71c -4.44141156513e-08
      92235.71c -0.0138774733425
      92236.71c -0.000291179645053
      43099.71c -3.91157149318e-05
      42095.71c -3.17885001417e-05
      61147.71c -1.61754141264e-05
      40090.71c -0.905641485227
      45103.71c -1.86378939567e-05
      55133.71c -5.72595431226e-05
      93237.71c -1.33849334499e-06
      55135.71c -3.95670895446e-05
      94240.71c -9.28162189491e-06
      94241.71c -6.88689778515e-07
      94242.71c -1.7691232034e-08
m4124 62152.71c -5.20336303452e-06
      62151.71c -1.9504374174e-06
      62150.71c -1.05758683247e-05
      64155.71c -6.85732337261e-09
      63153.71c -1.86971928525e-06
      60143.71c -5.50160145438e-05
      60145.71c -3.90017106933e-05
      95241.71c -1.83677492345e-08
      95243.71c -1.45925628625e-10
      94239.71c -0.000153067369553
      94238.71c -2.64124451551e-08
      63154.71c -6.37108747528e-08
      44103.71c -1.85161899418e-06
      92238.71c -0.0625630233459
      54135.71c -3.25201445931e-08
      62149.71c -6.15498183122e-07
      54131.71c -2.5338250609e-05
      1001.71c -0.0171233018595
      62147.71c -4.45783404321e-06
      92234.71c -4.67117915473e-08
      92235.71c -0.0137804217935
      92236.71c -0.000306846679039
      43099.71c -4.12516967867e-05
      42095.71c -3.35465068714e-05
      61147.71c -1.70253969907e-05
      40090.71c -0.905701861
      45103.71c -1.9654840061e-05
      55133.71c -6.03832951666e-05
      93237.71c -1.47410631331e-06
      55135.71c -4.09908097876e-05
      94240.71c -1.02536063709e-05
      94241.71c -8.00829327696e-07
      94242.71c -2.18231095192e-08
c **  D-13  Fuel ID= 3696
m4131 62152.71c -2.90766210838e-06
      62151.71c -1.5141218188e-06
      62150.71c -6.26830148389e-06
      64155.71c -6.69488307477e-09
      63153.71c -1.11599250736e-06
      60143.71c -3.42211438801e-05
      60145.71c -2.41753449002e-05
      95241.71c -1.47717722781e-08
      95243.71c -1.35392987039e-10
      94239.71c -8.46438676527e-05
      94238.71c -1.16795963157e-08
      63154.71c -2.79373997035e-08
      44103.71c -1.18049335912e-06
      92238.71c -0.0626219268193
      54135.71c -2.40377577021e-08
      62149.71c -6.21850380647e-07
      54131.71c -1.57032327966e-05
      1001.71c -0.0171149723901
      62147.71c -3.03990018626e-06
      92234.71c -2.5991694472e-08
      92235.71c -0.0144991349688
      92236.71c -0.000187543255016
      43099.71c -2.55296904693e-05
      42095.71c -2.05308308161e-05
      61147.71c -1.03903072852e-05
      40090.71c -0.905261290837
      45103.71c -1.21068706046e-05
      55133.71c -3.73700762771e-05
      93237.71c -6.27999645698e-07
      55135.71c -2.86519123728e-05
      94240.71c -4.12077318889e-06
      94241.71c -2.87600089718e-07
      94242.71c -1.25092726591e-08
m4132 62152.71c -3.78041511593e-06
      62151.71c -1.69874208681e-06
      62150.71c -7.94042789672e-06
      64155.71c -6.70837375172e-09
      63153.71c -1.43319923179e-06
      60143.71c -4.21919495229e-05
      60145.71c -2.99407407637e-05
      95241.71c -3.35766389119e-08
      95243.71c -5.68717316228e-10
      94239.71c -0.000117992752947
      94238.71c -2.66722657884e-08
      63154.71c -4.53593861352e-08
      44103.71c -1.47149276906e-06
      92238.71c -0.0625897015035
      54135.71c -2.81424477925e-08
      62149.71c -6.25579554634e-07
      54131.71c -1.94333717649e-05
      1001.71c -0.0171182331654
      62147.71c -3.73217709901e-06
      92234.71c -3.60763648529e-08
      92235.71c -0.0142186001195
      92236.71c -0.000235461493703
      43099.71c -3.1650138565e-05
      42095.71c -2.54689821193e-05
      61147.71c -1.27926059714e-05
      40090.71c -0.905433762844
      45103.71c -1.50141610755e-05
      55133.71c -4.63183166807e-05
      93237.71c -1.08126731568e-06
      55135.71c -3.37048600739e-05
      94240.71c -7.08788116891e-06
      94241.71c -6.68272316079e-07
      94242.71c -3.64356516061e-08
m4133 62152.71c -4.66390927142e-06
      62151.71c -1.82760749912e-06
      62150.71c -9.55731023348e-06
      64155.71c -6.57821118303e-09
      63153.71c -1.74095811781e-06
      60143.71c -4.97582663732e-05
      60145.71c -3.54723242656e-05
      95241.71c -5.1578087598e-08
      95243.71c -1.26244744137e-09
      94239.71c -0.000137016401958
      94238.71c -4.32862429538e-08
      63154.71c -6.34253569295e-08
      44103.71c -1.74549360422e-06
      92238.71c -0.0625727725134
      54135.71c -3.14452839205e-08
      62149.71c -6.18881750925e-07
      54131.71c -2.30047315735e-05
      1001.71c -0.0171213785426
      62147.71c -4.39620756256e-06
      92234.71c -4.22920013004e-08
      92235.71c -0.0139512924268
      92236.71c -0.00027867433625
      43099.71c -3.75272708313e-05
      42095.71c -3.02292688774e-05
      61147.71c -1.50928878285e-05
      40090.71c -0.905600131107
      45103.71c -1.77870210198e-05
      55133.71c -5.49080816892e-05
      93237.71c -1.4732549698e-06
      55135.71c -3.79095571789e-05
      94240.71c -9.66655790711e-06
      94241.71c -1.04710369709e-06
      94242.71c -6.81102196584e-08
m4134 62152.71c -4.97157938246e-06
      62151.71c -1.86424858502e-06
      62150.71c -1.01105990726e-05
      64155.71c -6.51881147999e-09
      63153.71c -1.84914096114e-06
      60143.71c -5.23161561545e-05
      60145.71c -3.7356508658e-05
      95241.71c -5.84424238366e-08
      95243.71c -1.60289311498e-09
      94239.71c -0.000143122449342
      94238.71c -5.01702441846e-08
      63154.71c -7.02304009234e-08
      44103.71c -1.84027849074e-06
      92238.71c -0.0625671290748
      54135.71c -3.24921792379e-08
      62149.71c -6.16465299961e-07
      54131.71c -2.42195610053e-05
      1001.71c -0.0171224546849
      62147.71c -4.62050359339e-06
      92234.71c -4.43352725191e-08
      92235.71c -0.0138603032927
      92236.71c -0.000293345407826
      43099.71c -3.95308852425e-05
      42095.71c -3.18511086657e-05
      61147.71c -1.58714886737e-05
      40090.71c -0.905657051439
      45103.71c -1.87298030659e-05
      55133.71c -5.78357373403e-05
      93237.71c -1.62051283061e-06
      55135.71c -3.92349336782e-05
      94240.71c -1.06128806214e-05
      94241.71c -1.1953868716e-06
      94242.71c -8.20806667225e-08
c **  D-14  Fuel ID= 3494
m4141 62152.71c -2.77296035796e-06
      62151.71c -1.44422925521e-06
      62150.71c -5.94226978692e-06
      64155.71c -6.43190623936e-09
      63153.71c -1.05194679363e-06
      60143.71c -3.26679820553e-05
      60145.71c -2.30422418728e-05
      95241.71c -6.0534321725e-09
      95243.71c -2.16682038337e-11
      94239.71c -8.21965306317e-05
      94238.71c -7.49471628821e-09
      63154.71c -2.49101245859e-08
      44103.71c -1.14589455558e-06
      92238.71c -0.0626235013509
      54135.71c -2.34957148572e-08
      62149.71c -6.23696262053e-07
      54131.71c -1.49693036471e-05
      1001.71c -0.0171143372956
      62147.71c -2.85655580328e-06
      92234.71c -2.51863619493e-08
      92235.71c -0.0145537459244
      92236.71c -0.000179066452314
      43099.71c -2.43243689074e-05
      42095.71c -1.94854138225e-05
      61147.71c -9.94873701802e-06
      40090.71c -0.905227698818
      45103.71c -1.15257521416e-05
      55133.71c -3.56041912807e-05
      93237.71c -5.66728737964e-07
      55135.71c -2.7410848704e-05
      94240.71c -3.76915828954e-06
      94241.71c -2.03282604879e-07
      94242.71c -4.47231493004e-09
m4142 62152.71c -3.67438605215e-06
      62151.71c -1.6242800002e-06
      62150.71c -7.64648719195e-06
      64155.71c -6.38885315315e-09
      63153.71c -1.36807951675e-06
      60143.71c -4.08313860521e-05
      60145.71c -2.89248956438e-05
      95241.71c -1.53660865499e-08
      95243.71c -1.04271616108e-10
      94239.71c -0.000115659137376
      94238.71c -1.74794770972e-08
      63154.71c -4.15187330591e-08
      44103.71c -1.44549798259e-06
      92238.71c -0.0625917362011
      54135.71c -2.77766668321e-08
      62149.71c -6.27792301644e-07
      54131.71c -1.87790649494e-05
      1001.71c -0.0171176578059
      62147.71c -3.56794214188e-06
      92234.71c -3.50088032028e-08
      92235.71c -0.0142676442904
      92236.71c -0.000227796507346
      43099.71c -3.05660322289e-05
      42095.71c -2.45096572186e-05
      61147.71c -1.24022347493e-05
      40090.71c -0.905403330387
      45103.71c -1.44922050929e-05
      55133.71c -4.47305722638e-05
      93237.71c -9.97223556319e-07
      55135.71c -3.25817841975e-05
      94240.71c -6.73267739221e-06
      94241.71c -5.15228800663e-07
      94242.71c -1.46003144408e-08
m4143 62152.71c -4.55863473191e-06
      62151.71c -1.74457507919e-06
      62150.71c -9.24816052218e-06
      64155.71c -6.22618521423e-09
      63153.71c -1.66859644498e-06
      60143.71c -4.83562656639e-05
      60145.71c -3.4405313754e-05
      95241.71c -2.51641401161e-08
      95243.71c -2.50438443634e-10
      94239.71c -0.000134319550029
      94238.71c -2.86837256027e-08
      63154.71c -5.87378771673e-08
      44103.71c -1.72068517835e-06
      92238.71c -0.06257540992
      54135.71c -3.11429827225e-08
      62149.71c -6.21505063855e-07
      54131.71c -2.23210007872e-05
      1001.71c -0.0171207680428
      62147.71c -4.23092747955e-06
      92234.71c -4.11759981894e-08
      92235.71c -0.0140028703378
      92236.71c -0.000270652986935
      43099.71c -3.63851813334e-05
      42095.71c -2.92094252139e-05
      61147.71c -1.46774050151e-05
      40090.71c -0.905567839973
      45103.71c -1.72406348681e-05
      55133.71c -5.32370827186e-05
      93237.71c -1.37309329407e-06
      55135.71c -3.67529606639e-05
      94240.71c -9.32418811143e-06
      94241.71c -8.43169075553e-07
      94242.71c -2.90033603461e-08
m4144 62152.71c -4.86248168743e-06
      62151.71c -1.77933193987e-06
      62150.71c -9.79199918554e-06
      64155.71c -6.17130467876e-09
      63153.71c -1.77380222875e-06
      60143.71c -5.08841918838e-05
      60145.71c -3.62574530093e-05
      95241.71c -2.92398228635e-08
      95243.71c -3.25507153694e-10
      94239.71c -0.000140286410999
      94238.71c -3.328038317e-08
      63154.71c -6.52203532079e-08
      44103.71c -1.8135049432e-06
      92238.71c -0.0625698277209
      54135.71c -3.21796910202e-08
      62149.71c -6.19207124373e-07
      54131.71c -2.35161326275e-05
      1001.71c -0.0171218270661
      62147.71c -4.4552302202e-06
      92234.71c -4.34100901803e-08
      92235.71c -0.0139134193751
      92236.71c -0.000285097557551
      43099.71c -3.83539787025e-05
      42095.71c -3.08030730241e-05
      61147.71c -1.54404080993e-05
      40090.71c -0.905623854835
      45103.71c -1.81686770559e-05
      55133.71c -5.61143143631e-05
      93237.71c -1.50976507939e-06
      55135.71c -3.80576709769e-05
      94240.71c -1.02611085393e-05
      94241.71c -9.79177568532e-07
      94242.71c -3.56990379999e-08
c **  D-15  Fuel ID= 3501
m4151 62152.71c -2.77286194246e-06
      62151.71c -1.56729202918e-06
      62150.71c -6.12915394899e-06
      64155.71c -6.97412788076e-09
      63153.71c -1.06572488819e-06
      60143.71c -3.3713634202e-05
      60145.71c -2.37105486629e-05
      95241.71c -2.9285850901e-09
      95243.71c -6.93879254848e-12
      94239.71c -8.55828192178e-05
      94238.71c -5.24469945943e-09
      63154.71c -2.21536060619e-08
      44103.71c -1.17960556831e-06
      92238.71c -0.0626228336281
      54135.71c -2.40309731196e-08
      62149.71c -6.23325821988e-07
      54131.71c -1.54105326867e-05
      1001.71c -0.0171146858404
      62147.71c -2.59155369173e-06
      92234.71c -2.587298166e-08
      92235.71c -0.0145213076448
      92236.71c -0.000184291424848
      43099.71c -2.50191690822e-05
      42095.71c -2.0003872573e-05
      61147.71c -1.06130314084e-05
      40090.71c -0.905246134377
      45103.71c -1.18637863407e-05
      55133.71c -3.6623868999e-05
      93237.71c -5.09332726212e-07
      55135.71c -2.81826891793e-05
      94240.71c -3.35677545006e-06
      94241.71c -1.3817965782e-07
      94242.71c -2.11462860819e-09
m4152 62152.71c -3.67076067346e-06
      62151.71c -1.76057956574e-06
      62150.71c -7.8400194032e-06
      64155.71c -6.93494377549e-09
      63153.71c -1.36992618569e-06
      60143.71c -4.19862150039e-05
      60145.71c -2.96299593875e-05
      95241.71c -7.48337483751e-09
      95243.71c -3.32638109408e-11
      94239.71c -0.000120295418985
      94238.71c -1.19480890004e-08
      63154.71c -3.65844582624e-08
      44103.71c -1.47910958116e-06
      92238.71c -0.0625909883642
      54135.71c -2.82421334543e-08
      62149.71c -6.26875822163e-07
      54131.71c -1.92506003984e-05
      1001.71c -0.0171180157675
      62147.71c -3.22904242549e-06
      92234.71c -3.61862667786e-08
      92235.71c -0.014233075907
      92236.71c -0.000233555630764
      43099.71c -3.12930821861e-05
      42095.71c -2.50544152414e-05
      61147.71c -1.31801934756e-05
      40090.71c -0.905422264026
      45103.71c -1.48521670612e-05
      55133.71c -4.58007023387e-05
      93237.71c -8.87064730597e-07
      55135.71c -3.33978841244e-05
      94240.71c -6.01033548097e-06
      94241.71c -3.51648146764e-07
      94242.71c -6.89226301534e-09
m4153 62152.71c -4.5707255779e-06
      62151.71c -1.88867091337e-06
      62150.71c -9.46850687899e-06
      64155.71c -6.73703991301e-09
      63153.71c -1.66220199171e-06
      60143.71c -4.9732912424e-05
      60145.71c -3.52197501282e-05
      95241.71c -1.24919665206e-08
      95243.71c -8.0532758994e-11
      94239.71c -0.000140436609274
      94238.71c -1.93598814249e-08
      63154.71c -5.170374535e-08
      44103.71c -1.75764304716e-06
      92238.71c -0.0625741989996
      54135.71c -3.15913136499e-08
      62149.71c -6.20459682322e-07
      54131.71c -2.28701951616e-05
      1001.71c -0.0171211751548
      62147.71c -3.83284986778e-06
      92234.71c -4.23641477989e-08
      92235.71c -0.0139629616855
      92236.71c -0.000277373990033
      43099.71c -3.72222959308e-05
      42095.71c -2.98450709109e-05
      61147.71c -1.55952528738e-05
      40090.71c -0.905589373324
      45103.71c -1.7663165324e-05
      55133.71c -5.447101941e-05
      93237.71c -1.21418509775e-06
      55135.71c -3.76802688045e-05
      94240.71c -8.4005942849e-06
      94241.71c -5.86261168422e-07
      94242.71c -1.38789016306e-08
m4154 62152.71c -4.88807420291e-06
      62151.71c -1.92450570039e-06
      62150.71c -1.00302075957e-05
      64155.71c -6.66177065673e-09
      63153.71c -1.76516899819e-06
      60143.71c -5.23826286273e-05
      60145.71c -3.71400616211e-05
      95241.71c -1.4559944666e-08
      95243.71c -1.05195551244e-10
      94239.71c -0.000146940944493
      94238.71c -2.24333052842e-08
      63154.71c -5.7469791512e-08
      44103.71c -1.85127582935e-06
      92238.71c -0.0625686971667
      54135.71c -3.2610410763e-08
      62149.71c -6.18165940819e-07
      54131.71c -2.41134925275e-05
      1001.71c -0.0171222636871
      62147.71c -4.04111044962e-06
      92234.71c -4.4585841904e-08
      92235.71c -0.0138702303535
      92236.71c -0.00029235374301
      43099.71c -3.92613084876e-05
      42095.71c -3.15020152288e-05
      61147.71c -1.64191627578e-05
      40090.71c -0.905646949005
      45103.71c -1.86308923731e-05
      55133.71c -5.74524623505e-05
      93237.71c -1.33656904079e-06
      55135.71c -3.90432717642e-05
      94240.71c -9.28676609978e-06
      94241.71c -6.82396573302e-07
      94242.71c -1.71373216663e-08
c **  D-17  Fuel ID= 3498
m4171 62152.71c -2.77757010193e-06
      62151.71c -1.5682205026e-06
      62150.71c -6.13802636096e-06
      64155.71c -6.9813535751e-09
      63153.71c -1.06723695963e-06
      60143.71c -3.37579078713e-05
      60145.71c -2.3741240411e-05
      95241.71c -2.94871738744e-09
      95243.71c -7.00896948072e-12
      94239.71c -8.56821682663e-05
      94238.71c -5.26727167942e-09
      63154.71c -2.22152658354e-08
      44103.71c -1.17964884708e-06
      92238.71c -0.062622764499
      54135.71c -2.40306469887e-08
      62149.71c -6.23281398754e-07
      54131.71c -1.54307889687e-05
      1001.71c -0.0171147026728
      62147.71c -2.59839851741e-06
      92234.71c -2.59056243048e-08
      92235.71c -0.0145198259888
      92236.71c -0.000184528563587
      43099.71c -2.50518520715e-05
      42095.71c -2.00371487553e-05
      61147.71c -1.06234396919e-05
      40090.71c -0.905247024694
      45103.71c -1.18805976858e-05
      55133.71c -3.66720298614e-05
      93237.71c -5.10609815949e-07
      55135.71c -2.82189942911e-05
      94240.71c -3.3661168189e-06
      94241.71c -1.38818717216e-07
      94242.71c -2.12972791407e-09
m4172 62152.71c -3.6768890493e-06
      62151.71c -1.76141370137e-06
      62150.71c -7.85105582939e-06
      64155.71c -6.94102225036e-09
      63153.71c -1.37192236341e-06
      60143.71c -4.20404709382e-05
      60145.71c -2.96678768625e-05
      95241.71c -7.53370792135e-09
      95243.71c -3.35948464534e-11
      94239.71c -0.000120428504889
      94238.71c -1.20006467075e-08
      63154.71c -3.66855096631e-08
      44103.71c -1.47914692084e-06
      92238.71c -0.0625908770972
      54135.71c -2.82410907475e-08
      62149.71c -6.2682164961e-07
      54131.71c -1.92755915442e-05
      1001.71c -0.0171180367218
      62147.71c -3.23749750265e-06
      92234.71c -3.62306606663e-08
      92235.71c -0.0142312452271
      92236.71c -0.000233851822162
      43099.71c -3.13335385859e-05
      42095.71c -2.5095758079e-05
      61147.71c -1.31928069034e-05
      40090.71c -0.905423372362
      45103.71c -1.48730021869e-05
      55133.71c -4.58602963519e-05
      93237.71c -8.89333249968e-07
      55135.71c -3.34404316443e-05
      94240.71c -6.02657377575e-06
      94241.71c -3.53230735208e-07
      94242.71c -6.94056258752e-09
m4173 62152.71c -4.57819470289e-06
      62151.71c -1.88937070333e-06
      62150.71c -9.4815361975e-06
      64155.71c -6.74224248259e-09
      63153.71c -1.66466123758e-06
      60143.71c -4.97961112466e-05
      60145.71c -3.52642860181e-05
      95241.71c -1.25745970142e-08
      95243.71c -8.13217558303e-11
      94239.71c -0.000140585493527
      94238.71c -1.94458384899e-08
      63154.71c -5.18451875153e-08
      44103.71c -1.75765041789e-06
      92238.71c -0.0625740653058
      54135.71c -3.15892357805e-08
      62149.71c -6.20396155129e-07
      54131.71c -2.28995138888e-05
      1001.71c -0.0171211999188
      62147.71c -3.84281303619e-06
      92234.71c -4.24156741821e-08
      92235.71c -0.0139608129176
      92236.71c -0.000277720998144
      43099.71c -3.72698874249e-05
      42095.71c -2.98939194229e-05
      61147.71c -1.56098353518e-05
      40090.71c -0.905590683168
      45103.71c -1.7687665338e-05
      55133.71c -5.45411095537e-05
      93237.71c -1.2173096123e-06
      55135.71c -3.77276944499e-05
      94240.71c -8.4227391035e-06
      94241.71c -5.88835733582e-07
      94242.71c -1.39747253758e-08
m4174 62152.71c -4.89599905555e-06
      62151.71c -1.92515749968e-06
      62150.71c -1.00439228955e-05
      64155.71c -6.66668591633e-09
      63153.71c -1.76779689216e-06
      60143.71c -5.24488193491e-05
      60145.71c -3.71868545712e-05
      95241.71c -1.46556546534e-08
      95243.71c -1.06219310505e-10
      94239.71c -0.000147094933506
      94238.71c -2.25333218212e-08
      63154.71c -5.76266624825e-08
      44103.71c -1.85128445487e-06
      92238.71c -0.062568554554
      54135.71c -3.26081419034e-08
      62149.71c -6.18098886155e-07
      54131.71c -2.41442770855e-05
      1001.71c -0.0171222897818
      62147.71c -4.05158488684e-06
      92234.71c -4.46392781059e-08
      92235.71c -0.013867972996
      92236.71c -0.000292718026076
      43099.71c -3.93113373259e-05
      42095.71c -3.15534039663e-05
      61147.71c -1.64344005857e-05
      40090.71c -0.90564832923
      45103.71c -1.86566295186e-05
      55133.71c -5.75261294641e-05
      93237.71c -1.34002606974e-06
      55135.71c -3.90922235515e-05
      94240.71c -9.31106783809e-06
      94241.71c -6.85373406074e-07
      94242.71c -1.72550999525e-08
c **  D-18  Fuel ID= 3336
m4181 62152.71c -2.89624311303e-06
      62151.71c -1.5133413917e-06
      62150.71c -6.24886737486e-06
      64155.71c -6.68566814145e-09
      63153.71c -1.11210317427e-06
      60143.71c -3.4127979985e-05
      60145.71c -2.41084515541e-05
      95241.71c -1.44649806381e-08
      95243.71c -1.32264427989e-10
      94239.71c -8.44649475594e-05
      94238.71c -1.1493707934e-08
      63154.71c -2.77004413506e-08
      44103.71c -1.18052038553e-06
      92238.71c -0.0626220873056
      54135.71c -2.40401351679e-08
      62149.71c -6.21956646743e-07
      54131.71c -1.56593549937e-05
      1001.71c -0.017114935107
      62147.71c -3.01783606985e-06
      92234.71c -2.59231909317e-08
      92235.71c -0.0145023607931
      92236.71c -0.000187030819908
      43099.71c -2.54581054479e-05
      42095.71c -2.04572217547e-05
      61147.71c -1.03754796884e-05
      40090.71c -0.905259318823
      45103.71c -1.20703434959e-05
      55133.71c -3.72646939866e-05
      93237.71c -6.23039211604e-07
      55135.71c -2.85736981634e-05
      94240.71c -4.08715802461e-06
      94241.71c -2.83125590195e-07
      94242.71c -1.22437247243e-08
m4182 62152.71c -3.7664072938e-06
      62151.71c -1.6981551471e-06
      62150.71c -7.91676887791e-06
      64155.71c -6.69994426719e-09
      63153.71c -1.42795948898e-06
      60143.71c -4.20816113201e-05
      60145.71c -2.98601374449e-05
      95241.71c -3.28906883957e-08
      95243.71c -5.55620976056e-10
      94239.71c -0.000117768471183
      94238.71c -2.6242985694e-08
      63154.71c -4.49788713069e-08
      44103.71c -1.4715699622e-06
      92238.71c -0.0625899592034
      54135.71c -2.81465630938e-08
      62149.71c -6.25705776338e-07
      54131.71c -1.93807097551e-05
      1001.71c -0.0171181876384
      62147.71c -3.70567329606e-06
      92234.71c -3.59846998524e-08
      92235.71c -0.0142224842466
      92236.71c -0.000234839581634
      43099.71c -3.1563606158e-05
      42095.71c -2.53796283567e-05
      61147.71c -1.27755051853e-05
      40090.71c -0.905431354789
      45103.71c -1.49700713689e-05
      55133.71c -4.61910249019e-05
      93237.71c -1.07267302921e-06
      55135.71c -3.3615300655e-05
      94240.71c -7.03404802169e-06
      94241.71c -6.58343067152e-07
      94242.71c -3.56714867089e-08
m4183 62152.71c -4.64760305078e-06
      62151.71c -1.82721511017e-06
      62150.71c -9.52987738151e-06
      64155.71c -6.57015960578e-09
      63153.71c -1.73446529724e-06
      60143.71c -4.96336271692e-05
      60145.71c -3.53795557578e-05
      95241.71c -5.05398875009e-08
      95243.71c -1.2334626565e-09
      94239.71c -0.000136780236817
      94238.71c -4.25878944436e-08
      63154.71c -6.29039036692e-08
      44103.71c -1.74565386713e-06
      92238.71c -0.0625730826072
      54135.71c -3.14514237048e-08
      62149.71c -6.19027709371e-07
      54131.71c -2.29443604557e-05
      1001.71c -0.0171213255815
      62147.71c -4.36560938715e-06
      92234.71c -4.21873028703e-08
      92235.71c -0.0139557615544
      92236.71c -0.000277961325157
      43099.71c -3.74274152366e-05
      42095.71c -3.0125678382e-05
      61147.71c -1.50739840051e-05
      40090.71c -0.905597329835
      45103.71c -1.77363882875e-05
      55133.71c -5.47612918387e-05
      93237.71c -1.46160896421e-06
      55135.71c -3.78117615937e-05
      94240.71c -9.59740141963e-06
      94241.71c -1.03216433909e-06
      94242.71c -6.6696700539e-08
m4184 62152.71c -4.9545332259e-06
      62151.71c -1.86391645484e-06
      62150.71c -1.00819049432e-05
      64155.71c -6.51078989555e-09
      63153.71c -1.84219875007e-06
      60143.71c -5.21869954663e-05
      60145.71c -3.72597308125e-05
      95241.71c -5.72727327124e-08
      95243.71c -1.56613315766e-09
      94239.71c -0.000142883972659
      94238.71c -4.93604597627e-08
      63154.71c -6.96567316377e-08
      44103.71c -1.84047302052e-06
      92238.71c -0.0625674568039
      54135.71c -3.24990757475e-08
      62149.71c -6.16617970536e-07
      54131.71c -2.41566613055e-05
      1001.71c -0.0171223992303
      62147.71c -4.58855310167e-06
      92234.71c -4.42266249543e-08
      92235.71c -0.0138649653827
      92236.71c -0.000292602836277
      43099.71c -3.94266227375e-05
      42095.71c -3.17427635786e-05
      61147.71c -1.58520398712e-05
      40090.71c -0.905654118284
      45103.71c -1.86770284138e-05
      55133.71c -5.7682492659e-05
      93237.71c -1.60771189367e-06
      55135.71c -3.91347135751e-05
      94240.71c -1.05384674318e-05
      94241.71c -1.1785887496e-06
      94242.71c -8.03840124101e-08
c **  E-2  Fuel ID= 2986
m5021 62152.71c -9.9615145621e-07
      62151.71c -8.35180182685e-07
      62150.71c -2.23721188708e-06
      64155.71c -2.65163539803e-09
      63153.71c -4.35790624929e-07
      60143.71c -1.37162362599e-05
      60145.71c -1.00360496371e-05
      95241.71c -2.63252906293e-10
      95243.71c -3.65843227925e-13
      94239.71c -3.75718434629e-05
      94238.71c -8.55449657983e-10
      63154.71c -5.1799239212e-09
      44103.71c -1.08357832875e-06
      92238.71c -0.0626521749098
      54135.71c -2.38986192675e-08
      62149.71c -5.95570348274e-07
      54131.71c -6.36944195789e-06
      1001.71c -0.0171072272352
      62147.71c -6.12801466804e-07
      92234.71c -1.11165182987e-08
      92235.71c -0.0151821173085
      92236.71c -7.82616416673e-05
      43099.71c -1.04913805118e-05
      42095.71c -6.19702136211e-06
      61147.71c -4.82494001275e-06
      40090.71c -0.904851626783
      45103.71c -4.44156401348e-06
      55133.71c -1.5210342709e-05
      93237.71c -1.4045808639e-07
      55135.71c -1.19480122032e-05
      94240.71c -7.84698656037e-07
      94241.71c -1.97015039282e-08
      94242.71c -1.81693089001e-10
m5022 62152.71c -1.31737171225e-06
      62151.71c -9.86959578886e-07
      62150.71c -2.94680828882e-06
      64155.71c -2.83848642489e-09
      63153.71c -5.55282335134e-07
      60143.71c -1.72028728181e-05
      60145.71c -1.26101940019e-05
      95241.71c -6.87762045372e-10
      95243.71c -1.80702826157e-12
      94239.71c -5.37632618628e-05
      94238.71c -1.89526161873e-09
      63154.71c -8.62601089535e-09
      44103.71c -1.36565454893e-06
      92238.71c -0.062638761674
      54135.71c -2.84052446466e-08
      62149.71c -6.19984647086e-07
      54131.71c -8.00251132703e-06
      1001.71c -0.0171086803858
      62147.71c -7.68059823587e-07
      92234.71c -1.55980921767e-08
      92235.71c -0.0150563391726
      92236.71c -9.98450145587e-05
      43099.71c -1.31892335924e-05
      42095.71c -7.79554474547e-06
      61147.71c -6.04256710928e-06
      40090.71c -0.9049284882
      45103.71c -5.58771943586e-06
      55133.71c -1.91198890251e-05
      93237.71c -2.36894844546e-07
      55135.71c -1.42308421255e-05
      94240.71c -1.43382237126e-06
      94241.71c -5.14204977234e-08
      94242.71c -6.0521948556e-10
m5023 62152.71c -1.6461970396e-06
      62151.71c -1.11091296312e-06
      62150.71c -3.64194175027e-06
      64155.71c -2.92689692601e-09
      63153.71c -6.70551375909e-07
      60143.71c -2.05301135576e-05
      60145.71c -1.50767954969e-05
      95241.71c -1.171435483e-09
      95243.71c -4.4565868354e-12
      94239.71c -6.37828966395e-05
      94238.71c -3.04785733888e-09
      63154.71c -1.23153679192e-08
      44103.71c -1.63456459365e-06
      92238.71c -0.0626316998661
      54135.71c -3.21478232846e-08
      62149.71c -6.2872442519e-07
      54131.71c -9.56538411906e-06
      1001.71c -0.0171100770556
      62147.71c -9.16800692221e-07
      92234.71c -1.85401400751e-08
      92235.71c -0.0149367936713
      92236.71c -0.000119338776137
      43099.71c -1.57751539438e-05
      42095.71c -9.33113969009e-06
      61147.71c -7.20672162123e-06
      40090.71c -0.905002362176
      45103.71c -6.68322909406e-06
      55133.71c -2.286657131e-05
      93237.71c -3.21346550363e-07
      55135.71c -1.61432029977e-05
      94240.71c -2.03721765849e-06
      94241.71c -8.75917385948e-08
      94242.71c -1.24395968059e-09
m5024 62152.71c -1.76485643062e-06
      62151.71c -1.15039487735e-06
      62150.71c -3.88644746858e-06
      64155.71c -2.9461737017e-09
      63153.71c -7.11354383638e-07
      60143.71c -2.16886884002e-05
      60145.71c -1.59383015651e-05
      95241.71c -1.37370935421e-09
      95243.71c -5.86147176625e-12
      94239.71c -6.70366618207e-05
      94238.71c -3.53158651995e-09
      63154.71c -1.37441186966e-08
      44103.71c -1.72872856648e-06
      92238.71c -0.0626294665667
      54135.71c -3.33631706173e-08
      62149.71c -6.30563725781e-07
      54131.71c -1.01108449027e-05
      1001.71c -0.0171105652539
      62147.71c -9.68733343193e-07
      92234.71c -1.95609035869e-08
      92235.71c -0.0148950848758
      92236.71c -0.000126123824188
      43099.71c -1.66785776246e-05
      42095.71c -9.86842180525e-06
      61147.71c -7.61193738637e-06
      40090.71c -0.905028184422
      45103.71c -7.06571618693e-06
      55133.71c -2.41753556889e-05
      93237.71c -3.53179897487e-07
      55135.71c -1.67633679969e-05
      94240.71c -2.26420626614e-06
      94241.71c -1.02647136221e-07
      94242.71c -1.54605765164e-09
c **  E-3  Fuel ID= 2458
m5031 62152.71c -2.53491273875e-06
      62151.71c -1.43941044654e-06
      62150.71c -5.55735719691e-06
      64155.71c -6.15954039865e-09
      63153.71c -9.78438320035e-07
      60143.71c -3.07752407997e-05
      60145.71c -2.17184726373e-05
      95241.71c -3.49176773999e-09
      95243.71c -1.0439460176e-11
      94239.71c -7.71937268724e-05
      94238.71c -5.25303518289e-09
      63154.71c -2.05931265386e-08
      44103.71c -1.1712882439e-06
      92238.71c -0.0626281233908
      54135.71c -2.3913951855e-08
      62149.71c -6.24904567818e-07
      54131.71c -1.40925125505e-05
      1001.71c -0.017113608589
      62147.71c -2.37621324488e-06
      92234.71c -2.33856146843e-08
      92235.71c -0.0146177984891
      92236.71c -0.000168564933377
      43099.71c -2.29063590837e-05
      42095.71c -1.78894123926e-05
      61147.71c -9.69181545628e-06
      40090.71c -0.905189155381
      45103.71c -1.07742171117e-05
      55133.71c -3.35099231593e-05
      93237.71c -4.68149871923e-07
      55135.71c -2.5734582983e-05
      94240.71c -3.08696716682e-06
      94241.71c -1.39922830577e-07
      94242.71c -2.58149611878e-09
m5032 62152.71c -3.29562759677e-06
      62151.71c -1.6161014034e-06
      62150.71c -7.02617022722e-06
      64155.71c -6.1818792305e-09
      63153.71c -1.2421141666e-06
      60143.71c -3.78596524054e-05
      60145.71c -2.68076119016e-05
      95241.71c -8.58118045463e-09
      95243.71c -4.69816970316e-11
      94239.71c -0.000108011458138
      94238.71c -1.16999302322e-08
      63154.71c -3.31956372216e-08
      44103.71c -1.45181242316e-06
      92238.71c -0.0625996216214
      54135.71c -2.79258527471e-08
      62149.71c -6.3008342871e-07
      54131.71c -1.73878232413e-05
      1001.71c -0.017116474633
      62147.71c -2.92137144829e-06
      92234.71c -3.23238472941e-08
      92235.71c -0.0143699003414
      92236.71c -0.000211077195167
      43099.71c -2.8298478646e-05
      42095.71c -2.21228032067e-05
      61147.71c -1.18934644062e-05
      40090.71c -0.905340748887
      45103.71c -1.33214054162e-05
      55133.71c -4.13912754243e-05
      93237.71c -8.00965753838e-07
      55135.71c -3.02142044669e-05
      94240.71c -5.4131772768e-06
      94241.71c -3.43764033115e-07
      94242.71c -8.00116134236e-09
m5033 62152.71c -4.08456558469e-06
      62151.71c -1.74419662264e-06
      62150.71c -8.47911273634e-06
      64155.71c -6.06969931111e-09
      63153.71c -1.50388766646e-06
      60143.71c -4.47499997876e-05
      60145.71c -3.17992436071e-05
      95241.71c -1.41348695405e-08
      95243.71c -1.12127249844e-10
      94239.71c -0.000125837193347
      94238.71c -1.89375593517e-08
      63154.71c -4.66920330752e-08
      44103.71c -1.72305173134e-06
      92238.71c -0.0625848453189
      54135.71c -3.12683798804e-08
      62149.71c -6.24409785405e-07
      54131.71c -2.06140556731e-05
      1001.71c -0.0171192995657
      62147.71c -3.45741816533e-06
      92234.71c -3.80782762156e-08
      92235.71c -0.0141286613753
      92236.71c -0.000250174541817
      43099.71c -3.35899239785e-05
      42095.71c -2.62898349402e-05
      61147.71c -1.40477951213e-05
      40090.71c -0.905490167896
      45103.71c -1.5807182189e-05
      55133.71c -4.91239130467e-05
      93237.71c -1.09425371826e-06
      55135.71c -3.40395843282e-05
      94240.71c -7.50456094462e-06
      94241.71c -5.65952350858e-07
      94242.71c -1.587370702e-08
m5034 62152.71c -4.36012508957e-06
      62151.71c -1.7810856575e-06
      62150.71c -8.97600004555e-06
      64155.71c -6.01926487319e-09
      63153.71c -1.59531138641e-06
      60143.71c -4.70851689349e-05
      60145.71c -3.34995640409e-05
      95241.71c -1.64146298327e-08
      95243.71c -1.45449988344e-10
      94239.71c -0.00013170849513
      94238.71c -2.19016075335e-08
      63154.71c -5.18047885693e-08
      44103.71c -1.81519232179e-06
      92238.71c -0.062579802052
      54135.71c -3.23073427979e-08
      62149.71c -6.22428873275e-07
      54131.71c -2.17121178525e-05
      1001.71c -0.0171202657652
      62147.71c -3.63971070398e-06
      92234.71c -3.9945053364e-08
      92235.71c -0.0140464924332
      92236.71c -0.000263490702308
      43099.71c -3.53941205632e-05
      42095.71c -2.77138669965e-05
      61147.71c -1.47775232726e-05
      40090.71c -0.905541273034
      45103.71c -1.66543038973e-05
      55133.71c -5.17599724568e-05
      93237.71c -1.20097991953e-06
      55135.71c -3.52492737339e-05
      94240.71c -8.2855789558e-06
      94241.71c -6.57121792043e-07
      94242.71c -1.95332064224e-08
c **  E-4  Fuel ID= 3006
m5041 62152.71c -2.66080330589e-06
      62151.71c -1.49725034354e-06
      62150.71c -5.8481698756e-06
      64155.71c -6.49567225662e-09
      63153.71c -1.03191189607e-06
      60143.71c -3.2207102639e-05
      60145.71c -2.27292271989e-05
      95241.71c -8.13947105101e-09
      95243.71c -6.77589862311e-11
      94239.71c -8.07759266794e-05
      94238.71c -7.6610130768e-09
      63154.71c -2.28147725846e-08
      44103.71c -1.18107762311e-06
      92238.71c -0.0626253962547
      54135.71c -2.40891543734e-08
      62149.71c -6.24147668678e-07
      54131.71c -1.47546710755e-05
      1001.71c -0.0171141663938
      62147.71c -2.56291239632e-06
      92234.71c -2.45107671744e-08
      92235.71c -0.0145688716736
      92236.71c -0.000176465299978
      43099.71c -2.39821469127e-05
      42095.71c -1.89395310398e-05
      61147.71c -1.00697604518e-05
      40090.71c -0.905218659317
      45103.71c -1.13172180958e-05
      55133.71c -3.50918945944e-05
      93237.71c -5.20763695173e-07
      55135.71c -2.6961057313e-05
      94240.71c -3.3940718509e-06
      94241.71c -1.90869198765e-07
      94242.71c -6.76858854739e-09
m5042 62152.71c -3.4775903849e-06
      62151.71c -1.68605347209e-06
      62150.71c -7.42896109313e-06
      64155.71c -6.52614296923e-09
      63153.71c -1.31992511306e-06
      60143.71c -3.98066296439e-05
      60145.71c -2.8198237316e-05
      95241.71c -1.87475827805e-08
      95243.71c -2.85597238394e-10
      94239.71c -0.000113144171618
      94238.71c -1.73919773935e-08
      63154.71c -3.71333179068e-08
      44103.71c -1.47316155032e-06
      92238.71c -0.0625952725278
      54135.71c -2.82314134431e-08
      62149.71c -6.28308248161e-07
      54131.71c -1.82949107798e-05
      1001.71c -0.0171172489517
      62147.71c -3.15921101401e-06
      92234.71c -3.40947271037e-08
      92235.71c -0.0143025681859
      92236.71c -0.000222016837328
      43099.71c -2.97794585518e-05
      42095.71c -2.35373093099e-05
      61147.71c -1.24229167431e-05
      40090.71c -0.905381704881
      45103.71c -1.40610183647e-05
      55133.71c -4.35664899985e-05
      93237.71c -8.95473802112e-07
      55135.71c -3.17687413319e-05
      94240.71c -5.92410223308e-06
      94241.71c -4.53619238051e-07
      94242.71c -1.99157367045e-08
m5043 62152.71c -4.31139716526e-06
      62151.71c -1.81912473311e-06
      62150.71c -8.96425972594e-06
      64155.71c -6.40415009832e-09
      63153.71c -1.6005946391e-06
      60143.71c -4.70637837421e-05
      60145.71c -3.34668306101e-05
      95241.71c -2.91340086852e-08
      95243.71c -6.35846846115e-10
      94239.71c -0.000131910922694
      94238.71c -2.81891637733e-08
      63154.71c -5.21524451312e-08
      44103.71c -1.74895820935e-06
      92238.71c -0.0625794762003
      54135.71c -3.15780153701e-08
      62149.71c -6.22037118449e-07
      54131.71c -2.1699613094e-05
      1001.71c -0.0171202336145
      62147.71c -3.73472827223e-06
      92234.71c -4.00286034898e-08
      92235.71c -0.0140479071858
      92236.71c -0.000263260277544
      43099.71c -3.53685666919e-05
      42095.71c -2.79898226863e-05
      61147.71c -1.46842200729e-05
      40090.71c -0.905539572486
      45103.71c -1.66924294818e-05
      55133.71c -5.17347406365e-05
      93237.71c -1.22148860104e-06
      55135.71c -3.57953868593e-05
      94240.71c -8.17151502794e-06
      94241.71c -7.24140782113e-07
      94242.71c -3.75523967996e-08
m5044 62152.71c -4.60307114979e-06
      62151.71c -1.85706850877e-06
      62150.71c -9.49028194252e-06
      64155.71c -6.34539878744e-09
      63153.71c -1.699062442e-06
      60143.71c -4.95239269039e-05
      60145.71c -3.52643400987e-05
      95241.71c -3.31557378041e-08
      95243.71c -8.08206825423e-10
      94239.71c -0.00013796699857
      94238.71c -3.26641145693e-08
      63154.71c -5.78286695475e-08
      44103.71c -1.84448388597e-06
      92238.71c -0.0625742140089
      54135.71c -3.26412697696e-08
      62149.71c -6.19765772042e-07
      54131.71c -2.28597789838e-05
      1001.71c -0.0171212558535
      62147.71c -3.92978957838e-06
      92234.71c -4.19865012896e-08
      92235.71c -0.0139610895632
      92236.71c -0.000277292303518
      43099.71c -3.72769113734e-05
      42095.71c -2.95088764775e-05
      61147.71c -1.54510394185e-05
      40090.71c -0.905593641717
      45103.71c -1.75889069522e-05
      55133.71c -5.45228540823e-05
      93237.71c -1.34377885661e-06
      55135.71c -3.70683495025e-05
      94240.71c -9.00419699337e-06
      94241.71c -8.32240714213e-07
      94242.71c -4.54019538412e-08
c **  E-5  Fuel ID= 5014
m5051 62152.71c -4.91411846634e-06
      62151.71c -1.90205013197e-06
      62150.71c -1.00330786903e-05
      64155.71c -9.6825977056e-09
      63153.71c -1.74924841259e-06
      60143.71c -5.30424453247e-05
      60145.71c -3.71989137253e-05
      95241.71c -1.65710809458e-08
      95243.71c -6.61120360194e-11
      94239.71c -0.000127809849854
      94238.71c -1.83385389581e-08
      63154.71c -5.30066412503e-08
      44103.71c -1.20620812863e-06
      92238.71c -0.0625922347427
      54135.71c -2.41319148211e-08
      62149.71c -6.02559209705e-07
      54131.71c -2.43001509391e-05
      1001.71c -0.0171221026779
      62147.71c -5.91619945353e-06
      92234.71c -4.01087494853e-08
      92235.71c -0.0138703481632
      92236.71c -0.000288352285563
      43099.71c -3.9392680331e-05
      42095.71c -3.46398778731e-05
      61147.71c -1.48378563535e-05
      40090.71c -0.905638432757
      45103.71c -1.92393883372e-05
      55133.71c -5.77963652168e-05
      93237.71c -1.14564833528e-06
      55135.71c -4.41115392506e-05
      94240.71c -8.00963303902e-06
      94241.71c -5.07027354931e-07
      94242.71c -1.2629426554e-08
m5052 62152.71c -6.43116319306e-06
      62151.71c -2.04648410338e-06
      62150.71c -1.26856766109e-05
      64155.71c -9.16105304053e-09
      63153.71c -2.27928021663e-06
      60143.71c -6.55677887035e-05
      60145.71c -4.62467465368e-05
      95241.71c -4.11137372734e-08
      95243.71c -3.0821468703e-10
      94239.71c -0.000176233048376
      94238.71c -4.25692805424e-08
      63154.71c -8.70921802986e-08
      44103.71c -1.50581204539e-06
      92238.71c -0.0625414401304
      54135.71c -2.80792249916e-08
      62149.71c -6.02050904744e-07
      54131.71c -3.01827907741e-05
      1001.71c -0.0171272363035
      62147.71c -7.32164567792e-06
      92234.71c -5.54628264533e-08
      92235.71c -0.0134312160371
      92236.71c -0.000363063685857
      43099.71c -4.90388663597e-05
      42095.71c -4.31879812849e-05
      61147.71c -1.82661948024e-05
      40090.71c -0.90590996533
      45103.71c -2.39652578899e-05
      55133.71c -7.19298736109e-05
      93237.71c -2.01653658722e-06
      55135.71c -5.2011718372e-05
      94240.71c -1.40044057101e-05
      94241.71c -1.25127573345e-06
      94242.71c -4.01287408113e-08
m5053 62152.71c -7.90638418049e-06
      62151.71c -2.11520946038e-06
      62150.71c -1.51784374924e-05
      64155.71c -8.61916260895e-09
      63153.71c -2.7870341525e-06
      60143.71c -7.70822695865e-05
      60145.71c -5.46886449915e-05
      95241.71c -6.70545816886e-08
      95243.71c -7.30664068519e-10
      94239.71c -0.000202269584168
      94238.71c -6.95531054621e-08
      63154.71c -1.22248537363e-07
      44103.71c -1.777258262e-06
      92238.71c -0.062514643972
      54135.71c -3.10397194966e-08
      62149.71c -5.91282732336e-07
      54131.71c -3.56573622904e-05
      1001.71c -0.0171320539495
      62147.71c -8.64069776439e-06
      92234.71c -6.45734141101e-08
      92235.71c -0.0130241417761
      92236.71c -0.000428682148745
      43099.71c -5.80501006558e-05
      42095.71c -5.12094763486e-05
      61147.71c -2.14414989666e-05
      40090.71c -0.906164784816
      45103.71c -2.83445719132e-05
      55133.71c -8.5128205774e-05
      93237.71c -2.76992400067e-06
      55135.71c -5.83692636652e-05
      94240.71c -1.92093599657e-05
      94241.71c -2.033717207e-06
      94242.71c -7.92344805331e-08
m5054 62152.71c -8.41500692572e-06
      62151.71c -2.13094397365e-06
      62150.71c -1.60323451655e-05
      64155.71c -8.4528614369e-09
      63153.71c -2.96729031184e-06
      60143.71c -8.09697676742e-05
      60145.71c -5.75637318994e-05
      95241.71c -7.75233865452e-08
      95243.71c -9.45039574708e-10
      94239.71c -0.000210492934883
      94238.71c -8.08224492379e-08
      63154.71c -1.35563981278e-07
      44103.71c -1.86750338528e-06
      92238.71c -0.0625056553269
      54135.71c -3.19061851673e-08
      62149.71c -5.87590444162e-07
      54131.71c -3.75189701149e-05
      1001.71c -0.0171337042455
      62147.71c -9.09057440431e-06
      92234.71c -6.76860467364e-08
      92235.71c -0.0128855759671
      92236.71c -0.000450920662447
      43099.71c -6.11234152838e-05
      42095.71c -5.39578970464e-05
      61147.71c -2.25101713262e-05
      40090.71c -0.906252073828
      45103.71c -2.98370841642e-05
      55133.71c -8.96279057803e-05
      93237.71c -3.05252188454e-06
      55135.71c -6.03674344359e-05
      94240.71c -2.11085404778e-05
      94241.71c -2.3482671067e-06
      94242.71c -9.71736631103e-08
c **  E-6  Fuel ID= 4744
m5061 62152.71c -4.76925319393e-06
      62151.71c -1.88732952516e-06
      62150.71c -9.78076649637e-06
      64155.71c -9.55237072082e-09
      63153.71c -1.70306880351e-06
      60143.71c -5.18085001015e-05
      60145.71c -3.63286301184e-05
      95241.71c -1.51144746116e-08
      95243.71c -5.85006450649e-11
      94239.71c -0.000125232177738
      94238.71c -1.71312197742e-08
      63154.71c -5.06241271161e-08
      44103.71c -1.20421293823e-06
      92238.71c -0.0625942751353
      54135.71c -2.41672254281e-08
      62149.71c -6.03929536382e-07
      54131.71c -2.37278154101e-05
      1001.71c -0.0171216212405
      62147.71c -5.67041445543e-06
      92234.71c -3.92040454756e-08
      92235.71c -0.0139123348187
      92236.71c -0.000281655146567
      43099.71c -3.846406387e-05
      42095.71c -3.36943619084e-05
      61147.71c -1.45998330281e-05
      40090.71c -0.905612968112
      45103.71c -1.87644408178e-05
      55133.71c -5.6429457877e-05
      93237.71c -1.09663168548e-06
      55135.71c -4.3086689189e-05
      94240.71c -7.65318822401e-06
      94241.71c -4.73454351727e-07
      94242.71c -1.14756149713e-08
m5062 62152.71c -6.24727364958e-06
      62151.71c -2.03539228723e-06
      62150.71c -1.23736861054e-05
      64155.71c -9.05680926579e-09
      63153.71c -2.21722059359e-06
      60143.71c -6.40744222771e-05
      60145.71c -4.5180059239e-05
      95241.71c -3.75744032855e-08
      95243.71c -2.73325957331e-10
      94239.71c -0.00017289164358
      94238.71c -3.9735519179e-08
      63154.71c -8.32001643003e-08
      44103.71c -1.50375574693e-06
      92238.71c -0.0625447378057
      54135.71c -2.81059630066e-08
      62149.71c -6.03757112753e-07
      54131.71c -2.9483353142e-05
      1001.71c -0.0171266397514
      62147.71c -7.0205862733e-06
      92234.71c -5.42392260804e-08
      92235.71c -0.0134826498399
      92236.71c -0.000354781203761
      43099.71c -4.78980704494e-05
      42095.71c -4.20217013436e-05
      61147.71c -1.79837102065e-05
      40090.71c -0.905878411934
      45103.71c -2.33814689665e-05
      55133.71c -7.02515734589e-05
      93237.71c -1.92981696849e-06
      55135.71c -5.08204365753e-05
      94240.71c -1.34019963768e-05
      94241.71c -1.17083014841e-06
      94242.71c -3.65251417709e-08
m5063 62152.71c -7.68727344842e-06
      62151.71c -2.10779313225e-06
      62150.71c -1.48123350989e-05
      64155.71c -8.53008523553e-09
      63153.71c -2.70992155547e-06
      60143.71c -7.53639921351e-05
      60145.71c -5.34452571131e-05
      95241.71c -6.13793707327e-08
      95243.71c -6.48688820619e-10
      94239.71c -0.000198657186307
      94238.71c -6.49030570376e-08
      63154.71c -1.16834313941e-07
      44103.71c -1.77579095997e-06
      92238.71c -0.0625186284702
      54135.71c -3.10766280061e-08
      62149.71c -5.93278294843e-07
      54131.71c -3.48438548695e-05
      1001.71c -0.0171313523523
      62147.71c -8.28855229498e-06
      92234.71c -6.31782641123e-08
      92235.71c -0.0130840538156
      92236.71c -0.000419059249128
      43099.71c -5.67174118589e-05
      42095.71c -4.98415922351e-05
      61147.71c -2.11208288813e-05
      40090.71c -0.906127675271
      45103.71c -2.76641190982e-05
      55133.71c -8.31685796568e-05
      93237.71c -2.65053913293e-06
      55135.71c -5.70522791453e-05
      94240.71c -1.84052613725e-05
      94241.71c -1.9062300799e-06
      94242.71c -7.22150530548e-08
m5064 62152.71c -8.184584241e-06
      62151.71c -2.12467997721e-06
      62150.71c -1.56483529261e-05
      64155.71c -8.36629619915e-09
      63153.71c -2.88487511645e-06
      60143.71c -7.91799652922e-05
      60145.71c -5.62628162633e-05
      95241.71c -7.10007673083e-08
      95243.71c -8.39591832408e-10
      94239.71c -0.0002068073976
      94238.71c -7.54183520197e-08
      63154.71c -1.29582904633e-07
      44103.71c -1.86657538223e-06
      92238.71c -0.0625098165557
      54135.71c -3.19731925643e-08
      62149.71c -5.89717002429e-07
      54131.71c -3.66684713577e-05
      1001.71c -0.017132969322
      62147.71c -8.72130218076e-06
      92234.71c -6.62329738507e-08
      92235.71c -0.0129482349224
      92236.71c -0.000440866123141
      43099.71c -5.97280269675e-05
      42095.71c -5.25226890979e-05
      61147.71c -2.21780684545e-05
      40090.71c -0.906213201559
      45103.71c -2.9124862124e-05
      55133.71c -8.75761107794e-05
      93237.71c -2.92146618667e-06
      55135.71c -5.90133446301e-05
      94240.71c -2.02339389223e-05
      94241.71c -2.2022458818e-06
      94242.71c -8.86128282493e-08
c **  E-7  Fuel ID= 3147
m5071 62152.71c -3.51477963458e-06
      62151.71c -1.68461977821e-06
      62150.71c -7.48365685323e-06
      64155.71c -7.9636995933e-09
      63153.71c -1.29955029401e-06
      60143.71c -4.04441355319e-05
      60145.71c -2.83897307825e-05
      95241.71c -6.61559588841e-09
      95243.71c -2.06094188556e-11
      94239.71c -9.91484299951e-05
      94238.71c -9.09094854592e-09
      63154.71c -3.23265526837e-08
      44103.71c -1.17931631092e-06
      92238.71c -0.0626137060535
      54135.71c -2.38479048062e-08
      62149.71c -6.15432758868e-07
      54131.71c -1.84987767769e-05
      1001.71c -0.0171172568741
      62147.71c -3.70062448178e-06
      92234.71c -3.04245796432e-08
      92235.71c -0.0142956726016
      92236.71c -0.000220105924795
      43099.71c -3.00054423576e-05
      42095.71c -2.51004797611e-05
      61147.71c -1.21310412403e-05
      40090.71c -0.905382123923
      45103.71c -1.44307578171e-05
      55133.71c -4.39724317491e-05
      93237.71c -7.10414443873e-07
      55135.71c -3.36138311627e-05
      94240.71c -4.84852405238e-06
      94241.71c -2.47444763148e-07
      94242.71c -4.91216620451e-09
m5072 62152.71c -4.56139818021e-06
      62151.71c -1.85141672712e-06
      62150.71c -9.39892429996e-06
      64155.71c -7.77513034676e-09
      63153.71c -1.65759202752e-06
      60143.71c -4.96246245671e-05
      60145.71c -3.49772853257e-05
      95241.71c -1.62223465272e-08
      95243.71c -9.26566597761e-11
      94239.71c -0.000137742862421
      94238.71c -2.03554444737e-08
      63154.71c -5.20352351952e-08
      44103.71c -1.45886772147e-06
      92238.71c -0.062576126203
      54135.71c -2.7720586315e-08
      62149.71c -6.18642698672e-07
      54131.71c -2.27785717127e-05
      1001.71c -0.0171209742463
      62147.71c -4.54197169028e-06
      92234.71c -4.1907318718e-08
      92235.71c -0.0139752463237
      92236.71c -0.000274962620724
      43099.71c -3.70061867568e-05
      42095.71c -3.09966414151e-05
      61147.71c -1.48363144403e-05
      40090.71c -0.905578746684
      45103.71c -1.78128930529e-05
      55133.71c -5.42212557933e-05
      93237.71c -1.2237128617e-06
      55135.71c -3.93949216704e-05
      94240.71c -8.45313741086e-06
      94241.71c -6.0539531336e-07
      94242.71c -1.51972780992e-08
m5073 62152.71c -5.63590209566e-06
      62151.71c -1.95700522498e-06
      62150.71c -1.12812770522e-05
      64155.71c -7.46923831153e-09
      63153.71c -2.01261472207e-06
      60143.71c -5.84887669267e-05
      60145.71c -4.14052036983e-05
      95241.71c -2.66680277162e-08
      95243.71c -2.20959604711e-10
      94239.71c -0.000159272984909
      94238.71c -3.30345835044e-08
      63154.71c -7.30296723218e-08
      44103.71c -1.72466676285e-06
      92238.71c -0.0625568126075
      54135.71c -3.08634834403e-08
      62149.71c -6.10917447934e-07
      54131.71c -2.69461141949e-05
      1001.71c -0.0171246192974
      62147.71c -5.36767492533e-06
      92234.71c -4.92392296115e-08
      92235.71c -0.0136649054221
      92236.71c -0.000325134916538
      43099.71c -4.38421416048e-05
      42095.71c -3.67805130784e-05
      61147.71c -1.74668889103e-05
      40090.71c -0.905771544169
      45103.71c -2.10946125048e-05
      55133.71c -6.42266393751e-05
      93237.71c -1.6766187097e-06
      55135.71c -4.4291361808e-05
      94240.71c -1.16583575605e-05
      94241.71c -9.92701884545e-07
      94242.71c -3.00985509993e-08
m5074 62152.71c -6.00990837877e-06
      62151.71c -1.98523485123e-06
      62150.71c -1.19260169252e-05
      64155.71c -7.35663355763e-09
      63153.71c -2.13736009654e-06
      60143.71c -6.14906605956e-05
      60145.71c -4.35967787568e-05
      95241.71c -3.09413418567e-08
      95243.71c -2.86447254694e-10
      94239.71c -0.000166424122207
      94238.71c -3.82405507084e-08
      63154.71c -8.09827392098e-08
      44103.71c -1.81574402278e-06
      92238.71c -0.0625500562091
      54135.71c -3.18426149874e-08
      62149.71c -6.08335501192e-07
      54131.71c -2.83651722932e-05
      1001.71c -0.0171258688033
      62147.71c -5.64791656865e-06
      92234.71c -5.16140400883e-08
      92235.71c -0.0135590889834
      92236.71c -0.000342238144979
      43099.71c -4.61751884791e-05
      42095.71c -3.87557011284e-05
      61147.71c -1.83576904083e-05
      40090.71c -0.905837634219
      45103.71c -2.22128479915e-05
      55133.71c -6.76403343789e-05
      93237.71c -1.84179959637e-06
      55135.71c -4.58389762263e-05
      94240.71c -1.2854902777e-05
      94241.71c -1.15067223525e-06
      94242.71c -3.70119727449e-08
c **  E-8  Fuel ID= 4991
m5081 62152.71c -4.96743848418e-06
      62151.71c -1.90693301075e-06
      62150.71c -1.01251628488e-05
      64155.71c -9.72686557463e-09
      63153.71c -1.76629361358e-06
      60143.71c -5.34921482704e-05
      60145.71c -3.75164290672e-05
      95241.71c -1.71708370173e-08
      95243.71c -6.94338083017e-11
      94239.71c -0.000128738040308
      94238.71c -1.88184990786e-08
      63154.71c -5.39115509965e-08
      44103.71c -1.20683093017e-06
      92238.71c -0.0625915016695
      54135.71c -2.41247583028e-08
      62149.71c -6.02082669013e-07
      54131.71c -2.45091037129e-05
      1001.71c -0.0171222781259
      62147.71c -6.00836469715e-06
      92234.71c -4.04359482537e-08
      92235.71c -0.0138550354053
      92236.71c -0.000290794467622
      43099.71c -3.97316961312e-05
      42095.71c -3.49849022692e-05
      61147.71c -1.49221752053e-05
      40090.71c -0.905647712716
      45103.71c -1.94127417297e-05
      55133.71c -5.8295678574e-05
      93237.71c -1.16429594931e-06
      55135.71c -4.44850742467e-05
      94240.71c -8.14456174019e-06
      94241.71c -5.2029544542e-07
      94242.71c -1.31088927497e-08
m5082 62152.71c -6.49856288776e-06
      62151.71c -2.05005749813e-06
      62150.71c -1.27994053521e-05
      64155.71c -9.19702275901e-09
      63153.71c -2.30223105541e-06
      60143.71c -6.61109417588e-05
      60145.71c -4.66352764152e-05
      95241.71c -4.25639591709e-08
      95243.71c -3.23334543159e-10
      94239.71c -0.000177428771327
      94238.71c -4.36959393022e-08
      63154.71c -8.85692876113e-08
      44103.71c -1.50635758883e-06
      92238.71c -0.0625402668764
      54135.71c -2.80609180612e-08
      62149.71c -6.01455846214e-07
      54131.71c -3.04378229376e-05
      1001.71c -0.0171274531018
      62147.71c -7.43438259072e-06
      92234.71c -5.59061827706e-08
      92235.71c -0.0134124915523
      92236.71c -0.000366077901489
      43099.71c -4.94548678946e-05
      42095.71c -4.36130332606e-05
      61147.71c -1.83659702986e-05
      40090.71c -0.905921432432
      45103.71c -2.41779882409e-05
      55133.71c -7.25421814586e-05
      93237.71c -2.0493719888e-06
      55135.71c -5.24452639467e-05
      94240.71c -1.42314172786e-05
      94241.71c -1.28283963709e-06
      94242.71c -4.16196737044e-08
m5083 62152.71c -7.98633606696e-06
      62151.71c -2.11749634639e-06
      62150.71c -1.53116075894e-05
      64155.71c -8.65092286479e-09
      63153.71c -2.81556045618e-06
      60143.71c -7.77055758906e-05
      60145.71c -5.51405619244e-05
      95241.71c -6.93707997991e-08
      95243.71c -7.66069561511e-10
      94239.71c -0.000203556617209
      94238.71c -7.14018900528e-08
      63154.71c -1.24300372928e-07
      44103.71c -1.77742929501e-06
      92238.71c -0.0625132124577
      54135.71c -3.10099501263e-08
      62149.71c -5.90616967544e-07
      54131.71c -3.5953296486e-05
      1001.71c -0.0171323087639
      62147.71c -8.77240536933e-06
      92234.71c -6.5080330555e-08
      92235.71c -0.0130023722898
      92236.71c -0.000432175927719
      43099.71c -5.85350077864e-05
      42095.71c -5.17074452409e-05
      61147.71c -2.15541525091e-05
      40090.71c -0.9061782627
      45103.71c -2.85920638822e-05
      55133.71c -8.58417423266e-05
      93237.71c -2.81514257632e-06
      55135.71c -5.88476023061e-05
      94240.71c -1.95109987566e-05
      94241.71c -2.08349495638e-06
      94242.71c -8.21267032793e-08
m5084 62152.71c -8.49909160466e-06
      62151.71c -2.13281108142e-06
      62150.71c -1.61720584687e-05
      64155.71c -8.4830929954e-09
      63153.71c -2.99779304446e-06
      60143.71c -8.16187316424e-05
      60145.71c -5.80367090481e-05
      95241.71c -8.01821275481e-08
      95243.71c -9.90623007885e-10
      94239.71c -0.000211800651557
      94238.71c -8.29696950784e-08
      63154.71c -1.37827940964e-07
      44103.71c -1.86766622174e-06
      92238.71c -0.0625041419566
      54135.71c -3.18754997907e-08
      62149.71c -5.86889197475e-07
      54131.71c -3.78283580782e-05
      1001.71c -0.0171339717907
      62147.71c -9.22861844254e-06
      92234.71c -6.82101871144e-08
      92235.71c -0.0128627963439
      92236.71c -0.000454571789164
      43099.71c -6.16313091024e-05
      42095.71c -5.44800855409e-05
      61147.71c -2.26268502879e-05
      40090.71c -0.906266225086
      45103.71c -3.00959866259e-05
      55133.71c -9.0375050339e-05
      93237.71c -3.10211791272e-06
      55135.71c -6.08592465594e-05
      94240.71c -2.1436514368e-05
      94241.71c -2.405255036e-06
      94242.71c -1.00700718241e-07
c **  E-9  Fuel ID= 4742
m5091 62152.71c -8.06255201822e-06
      62151.71c -2.08064707114e-06
      62150.71c -1.53199822046e-05
      64155.71c -1.15792876306e-08
      63153.71c -2.78504018535e-06
      60143.71c -7.84717576981e-05
      60145.71c -5.5397485283e-05
      95241.71c -7.64702230151e-08
      95243.71c -5.21222855356e-10
      94239.71c -0.00017731487383
      94238.71c -5.94510410016e-08
      63154.71c -1.1483726636e-07
      44103.71c -1.24058361509e-06
      92238.71c -0.0625486743733
      54135.71c -2.40715432671e-08
      62149.71c -5.73373647907e-07
      54131.71c -3.62428143267e-05
      1001.71c -0.0171322294821
      62147.71c -1.18132500034e-05
      92234.71c -5.88147355961e-08
      92235.71c -0.0129929925367
      92236.71c -0.000427928175267
      43099.71c -5.88533668404e-05
      42095.71c -5.4467988859e-05
      61147.71c -1.89836319039e-05
      40090.71c -0.906174069248
      45103.71c -2.91547218173e-05
      55133.71c -8.64408203482e-05
      93237.71c -2.42315590255e-06
      55135.71c -6.5433092526e-05
      94240.71c -1.70568384936e-05
      94241.71c -1.58118196402e-06
      94242.71c -6.32805726259e-08
m5092 62152.71c -1.03405394967e-05
      62151.71c -2.15770866777e-06
      62150.71c -1.91809045953e-05
      64155.71c -1.07756836087e-08
      63153.71c -3.68726290029e-06
      60143.71c -9.59692767807e-05
      60145.71c -6.83739276658e-05
      95241.71c -1.8185096416e-07
      95243.71c -2.33345890487e-09
      94239.71c -0.000238391151526
      94238.71c -1.39253143473e-07
      63154.71c -1.87764864788e-07
      44103.71c -1.53753811174e-06
      92238.71c -0.062471272343
      54135.71c -2.75656001958e-08
      62149.71c -5.66972847822e-07
      54131.71c -4.46516999087e-05
      1001.71c -0.0171397099359
      62147.71c -1.44833009666e-05
      92234.71c -8.04675920536e-08
      92235.71c -0.0123654485231
      92236.71c -0.000533981884226
      43099.71c -7.27775080349e-05
      42095.71c -6.74710036775e-05
      61147.71c -2.30852647856e-05
      40090.71c -0.906569732481
      45103.71c -3.60465821593e-05
      55133.71c -0.000106844992948
      93237.71c -4.25923063944e-06
      55135.71c -7.65974910895e-05
      94240.71c -2.88725109321e-05
      94241.71c -3.73605069937e-06
      94242.71c -1.93902862243e-07
m5093 62152.71c -1.24748971544e-05
      62151.71c -2.16697403028e-06
      62150.71c -2.27411051869e-05
      64155.71c -1.01334890687e-08
      63153.71c -4.53996079939e-06
      60143.71c -0.000111585499432
      60145.71c -8.02554234372e-05
      95241.71c -2.86389127395e-07
      95243.71c -5.35997063986e-09
      94239.71c -0.000267367003503
      94238.71c -2.27983889309e-07
      63154.71c -2.6121227611e-07
      44103.71c -1.79449902153e-06
      92238.71c -0.0624301648986
      54135.71c -2.99775181092e-08
      62149.71c -5.50480650211e-07
      54131.71c -5.23177590263e-05
      1001.71c -0.0171466198871
      62147.71c -1.69538909026e-05
      92234.71c -9.29392492392e-08
      92235.71c -0.0117933459604
      92236.71c -0.000625376850375
      43099.71c -8.55506685529e-05
      42095.71c -7.94724037394e-05
      61147.71c -2.67843355145e-05
      40090.71c -0.906935220149
      45103.71c -4.22839148902e-05
      55133.71c -0.000125552403814
      93237.71c -5.82532201749e-06
      55135.71c -8.53041706818e-05
      94240.71c -3.86150759828e-05
      94241.71c -5.85080168983e-06
      94242.71c -3.71668729138e-07
m5094 62152.71c -1.31957307595e-05
      62151.71c -2.16509640488e-06
      62150.71c -2.39499723316e-05
      64155.71c -9.9584295429e-09
      63153.71c -4.84174109545e-06
      60143.71c -0.000116759687112
      60145.71c -8.42575892899e-05
      95241.71c -3.27261899856e-07
      95243.71c -6.86438092675e-09
      94239.71c -0.000275997972349
      94238.71c -2.64725015368e-07
      63154.71c -2.88647759512e-07
      44103.71c -1.8795403073e-06
      92238.71c -0.0624166271396
      54135.71c -3.0672207683e-08
      62149.71c -5.4472518724e-07
      54131.71c -5.48916330846e-05
      1001.71c -0.0171489630508
      62147.71c -1.77851407357e-05
      92234.71c -9.70637242226e-08
      92235.71c -0.0116007456226
      92236.71c -0.000655959464935
      43099.71c -8.98614335502e-05
      42095.71c -8.35388054706e-05
      61147.71c -2.80061499872e-05
      40090.71c -0.907059156975
      45103.71c -4.43804451679e-05
      55133.71c -0.000131861529974
      93237.71c -6.40673456291e-06
      55135.71c -8.79918655239e-05
      94240.71c -4.20815457813e-05
      94241.71c -6.67377254599e-06
      94242.71c -4.51442907099e-07
c **  E-10  Fuel ID= 4351
m5101 62152.71c -6.40625343898e-06
      62151.71c -2.00746440065e-06
      62150.71c -1.25689576838e-05
      64155.71c -1.08073629405e-08
      63153.71c -2.23724676882e-06
      60143.71c -6.53173851858e-05
      60145.71c -4.59326351591e-05
      95241.71c -4.1493661206e-08
      95243.71c -2.41776990949e-10
      94239.71c -0.000153039584406
      94238.71c -3.60543832703e-08
      63154.71c -8.10596563261e-08
      44103.71c -1.21377531522e-06
      92238.71c -0.0625704314756
      54135.71c -2.39759792774e-08
      62149.71c -5.89155468259e-07
      54131.71c -3.00384024659e-05
      1001.71c -0.0171269517671
      62147.71c -8.6935556897e-06
      92234.71c -4.93773415314e-08
      92235.71c -0.0134490405366
      92236.71c -0.000355649000008
      43099.71c -4.872698816e-05
      42095.71c -4.42019133993e-05
      61147.71c -1.68911197716e-05
      40090.71c -0.905894915364
      45103.71c -2.40105238402e-05
      55133.71c -7.15398425168e-05
      93237.71c -1.73834817826e-06
      55135.71c -5.44473554551e-05
      94240.71c -1.21594666009e-05
      94241.71c -9.75801124693e-07
      94242.71c -3.3071072513e-08
m5102 62152.71c -8.35164864124e-06
      62151.71c -2.11988347066e-06
      62150.71c -1.59002404046e-05
      64155.71c -1.00789710758e-08
      63153.71c -2.96135970966e-06
      60143.71c -8.07268839377e-05
      60145.71c -5.72067525355e-05
      95241.71c -1.01638052005e-07
      95243.71c -1.12169358641e-09
      94239.71c -0.000209177052694
      94238.71c -8.53091094644e-08
      63154.71c -1.34318734875e-07
      44103.71c -1.51860641152e-06
      92238.71c -0.0625057386716
      54135.71c -2.77894487787e-08
      62149.71c -5.85325239346e-07
      54131.71c -3.73582214256e-05
      1001.71c -0.0171333966415
      62147.71c -1.07642418509e-05
      92234.71c -6.80026137624e-08
      92235.71c -0.0129030667325
      92236.71c -0.000448037740955
      43099.71c -6.07881992772e-05
      42095.71c -5.5232111009e-05
      61147.71c -2.07743495278e-05
      40090.71c -0.906235803748
      45103.71c -2.99598021684e-05
      55133.71c -8.9215615997e-05
      93237.71c -3.08627857782e-06
      55135.71c -6.42250672854e-05
      94240.71c -2.10916056408e-05
      94241.71c -2.38022651416e-06
      94242.71c -1.04734958192e-07
m5103 62152.71c -1.01864621798e-05
      62151.71c -2.15693344529e-06
      62150.71c -1.89696564422e-05
      64155.71c -9.45526780378e-09
      63153.71c -3.64452182917e-06
      60143.71c -9.45388338419e-05
      60145.71c -6.75182117353e-05
      95241.71c -1.61898385887e-07
      95243.71c -2.60879483146e-09
      94239.71c -0.000237057680623
      94238.71c -1.40469638361e-07
      63154.71c -1.88287966736e-07
      44103.71c -1.78238295828e-06
      92238.71c -0.0624720725385
      54135.71c -3.04766338252e-08
      62149.71c -5.71600692695e-07
      54131.71c -4.40292456156e-05
      1001.71c -0.0171393377358
      62147.71c -1.26732737592e-05
      92234.71c -7.90648761561e-08
      92235.71c -0.0124061091499
      92236.71c -0.000527768320862
      43099.71c -7.18351701082e-05
      42095.71c -6.54049515333e-05
      61147.71c -2.42824885584e-05
      40090.71c -0.906550045719
      45103.71c -3.53520012817e-05
      55133.71c -0.00010539823816
      93237.71c -4.25275686542e-06
      55135.71c -7.18755128281e-05
      94240.71c -2.85381695183e-05
      94241.71c -3.78331464261e-06
      94242.71c -2.02867513837e-07
m5104 62152.71c -1.08000978747e-05
      62151.71c -2.16276132459e-06
      62150.71c -1.99970400018e-05
      64155.71c -9.26788892558e-09
      63153.71c -3.88271354829e-06
      60143.71c -9.90683689545e-05
      60145.71c -7.09448980493e-05
      95241.71c -1.8605746451e-07
      95243.71c -3.35540985829e-09
      94239.71c -0.000245407197687
      94238.71c -1.6283950034e-07
      63154.71c -2.08329036111e-07
      44103.71c -1.87088425854e-06
      92238.71c -0.0624609264469
      54135.71c -3.12790445052e-08
      62149.71c -5.66880252599e-07
      54131.71c -4.62394312474e-05
      1001.71c -0.0171413284497
      62147.71c -1.33074210269e-05
      92234.71c -8.29559642104e-08
      92235.71c -0.0122410265825
      92236.71c -0.000554150464826
      43099.71c -7.5513350109e-05
      42095.71c -6.87963951779e-05
      61147.71c -2.54304244798e-05
      40090.71c -0.906655340444
      45103.71c -3.71376790381e-05
      55133.71c -0.000110782933844
      93237.71c -4.6680825303e-06
      55135.71c -7.42137824594e-05
      94240.71c -3.11670158272e-05
      94241.71c -4.33861933303e-06
      94242.71c -2.47550960703e-07
c **  E-11  Fuel ID= 3107
m5111 62152.71c -4.03142607814e-06
      62151.71c -1.59092505943e-06
      62150.71c -8.18084727604e-06
      64155.71c -7.60174131751e-09
      63153.71c -1.49874880599e-06
      60143.71c -4.33895711654e-05
      60145.71c -3.07584399827e-05
      95241.71c -4.49636920336e-08
      95243.71c -4.43280137397e-10
      94239.71c -0.0001022517174
      94238.71c -2.99732079183e-08
      63154.71c -5.12568994689e-08
      44103.71c -1.17783364104e-06
      92238.71c -0.0626061330681
      54135.71c -2.38037869999e-08
      62149.71c -6.11392542029e-07
      54131.71c -2.00213266811e-05
      1001.71c -0.0171186414902
      62147.71c -5.21126975843e-06
      92234.71c -3.27332516783e-08
      92235.71c -0.014181675776
      92236.71c -0.00023797292009
      43099.71c -3.2574501807e-05
      42095.71c -2.77748315794e-05
      61147.71c -1.18495178884e-05
      40090.71c -0.905455360345
      45103.71c -1.57015693923e-05
      55133.71c -4.77409379432e-05
      93237.71c -1.11616492985e-06
      55135.71c -3.63491140735e-05
      94240.71c -7.42890225771e-06
      94241.71c -7.27943674471e-07
      94242.71c -3.86423248389e-08
m5112 62152.71c -5.15895020744e-06
      62151.71c -1.75650388385e-06
      62150.71c -1.02687546965e-05
      64155.71c -7.5379345593e-09
      63153.71c -1.94885179124e-06
      60143.71c -5.30505314706e-05
      60145.71c -3.78730590446e-05
      95241.71c -1.01082268924e-07
      95243.71c -1.85755182154e-09
      94239.71c -0.00014006472667
      94238.71c -6.89184957065e-08
      63154.71c -8.28065380889e-08
      44103.71c -1.46389605291e-06
      92238.71c -0.0625643407857
      54135.71c -2.77374535541e-08
      62149.71c -6.13157848624e-07
      54131.71c -2.46159353321e-05
      1001.71c -0.0171227135563
      62147.71c -6.34046424832e-06
      92234.71c -4.50972832127e-08
      92235.71c -0.0138363561536
      92236.71c -0.000296664984389
      43099.71c -4.01659491281e-05
      42095.71c -3.42624473821e-05
      61147.71c -1.44755252374e-05
      40090.71c -0.905670743914
      45103.71c -1.93531087882e-05
      55133.71c -5.88453306878e-05
      93237.71c -1.92704658001e-06
      55135.71c -4.25185643518e-05
      94240.71c -1.23856984785e-05
      94241.71c -1.64542764801e-06
      94242.71c -1.11638500209e-07
m5113 62152.71c -6.26863391487e-06
      62151.71c -1.86622320457e-06
      62150.71c -1.22570267755e-05
      64155.71c -7.37058145122e-09
      63153.71c -2.37992832447e-06
      60143.71c -6.2024235694e-05
      60145.71c -4.46018407702e-05
      95241.71c -1.53749235487e-07
      95243.71c -4.11489250421e-09
      94239.71c -0.000160257840334
      94238.71c -1.12011981962e-07
      63154.71c -1.14742514695e-07
      44103.71c -1.72972184169e-06
      92238.71c -0.0625422556282
      54135.71c -3.08410566488e-08
      62149.71c -6.04517716511e-07
      54131.71c -2.89459623854e-05
      1001.71c -0.0171265905463
      62147.71c -7.40742930444e-06
      92234.71c -5.25955630257e-08
      92235.71c -0.0135114775038
      92236.71c -0.00034884304657
      43099.71c -4.73542403735e-05
      42095.71c -4.04237967464e-05
      61147.71c -1.69532472513e-05
      40090.71c -0.905875809329
      45103.71c -2.27698797216e-05
      55133.71c -6.93539361734e-05
      93237.71c -2.61935942774e-06
      55135.71c -4.75337974619e-05
      94240.71c -1.64723728261e-05
      94241.71c -2.51731291692e-06
      94242.71c -2.07217190838e-07
m5114 62152.71c -6.64912245674e-06
      62151.71c -1.8969341158e-06
      62150.71c -1.2934440206e-05
      64155.71c -7.30823010239e-09
      63153.71c -2.53233649945e-06
      60143.71c -6.50270929149e-05
      60145.71c -4.68805913399e-05
      95241.71c -1.73553841342e-07
      95243.71c -5.22050695121e-09
      94239.71c -0.000166591370717
      94238.71c -1.29862592658e-07
      63154.71c -1.26686232121e-07
      44103.71c -1.82113446353e-06
      92238.71c -0.0625348766552
      54135.71c -3.18134812694e-08
      62149.71c -6.01440712749e-07
      54131.71c -3.04096341446e-05
      1001.71c -0.0171279120698
      62147.71c -7.76480922742e-06
      92234.71c -5.50274757316e-08
      92235.71c -0.0134014985902
      92236.71c -0.00036642321604
      43099.71c -4.97915467919e-05
      42095.71c -4.25135445014e-05
      61147.71c -1.77854804755e-05
      40090.71c -0.905945708602
      45103.71c -2.39234519575e-05
      55133.71c -7.29168233856e-05
      93237.71c -2.88027616921e-06
      55135.71c -4.90977751469e-05
      94240.71c -1.79360170934e-05
      94241.71c -2.84852041725e-06
      94242.71c -2.49051483958e-07
c **  E-12  Fuel ID= 3690
m5121 62152.71c -2.49025276365e-06
      62151.71c -1.49455036603e-06
      62150.71c -5.5705607097e-06
      64155.71c -6.44171945863e-09
      63153.71c -9.73050581524e-07
      60143.71c -3.09057496277e-05
      60145.71c -2.17759073651e-05
      95241.71c -2.14065420422e-09
      95243.71c -4.65634707208e-12
      94239.71c -7.90913520729e-05
      94238.71c -4.19843617193e-09
      63154.71c -1.88354862277e-08
      44103.71c -1.17652953984e-06
      92238.71c -0.0626271155849
      54135.71c -2.40817740779e-08
      62149.71c -6.26053583426e-07
      54131.71c -1.41322230597e-05
      1001.71c -0.0171136283906
      62147.71c -2.21345982966e-06
      92234.71c -2.37961026909e-08
      92235.71c -0.0146147427116
      92236.71c -0.000169320093611
      43099.71c -2.2960526149e-05
      42095.71c -1.79152175094e-05
      61147.71c -9.89881417958e-06
      40090.71c -0.905190202746
      45103.71c -1.08038300433e-05
      55133.71c -3.3589490285e-05
      93237.71c -4.39998148693e-07
      55135.71c -2.58901412561e-05
      94240.71c -2.85291192148e-06
      94241.71c -1.08820759281e-07
      94242.71c -1.53502234592e-09
m5122 62152.71c -3.29931385056e-06
      62151.71c -1.69123220154e-06
      62150.71c -7.14354887295e-06
      64155.71c -6.47233970315e-09
      63153.71c -1.24847791867e-06
      60143.71c -3.8528071978e-05
      60145.71c -2.72318148015e-05
      95241.71c -5.49181345829e-09
      95243.71c -2.23925477974e-11
      94239.71c -0.000111486228233
      94238.71c -9.52601432524e-09
      63154.71c -3.11307156167e-08
      44103.71c -1.47638635495e-06
      92238.71c -0.0625979414157
      54135.71c -2.8349240307e-08
      62149.71c -6.30144049405e-07
      54131.71c -1.76673291798e-05
      1001.71c -0.0171166954138
      62147.71c -2.76046010071e-06
      92234.71c -3.33493434351e-08
      92235.71c -0.0143489441591
      92236.71c -0.000214776984819
      43099.71c -2.87369067331e-05
      42095.71c -2.24516920892e-05
      61147.71c -1.2308190751e-05
      40090.71c -0.905352426634
      45103.71c -1.35340001783e-05
      55133.71c -4.20338359768e-05
      93237.71c -7.63777262948e-07
      55135.71c -3.07022608241e-05
      94240.71c -5.124211738e-06
      94241.71c -2.78146465966e-07
      94242.71c -5.02085829163e-09
m5123 62152.71c -4.11318148618e-06
      62151.71c -1.82643271324e-06
      62150.71c -8.64323821479e-06
      64155.71c -6.33732379156e-09
      63153.71c -1.51302586944e-06
      60143.71c -4.56820881063e-05
      60145.71c -3.2390652212e-05
      95241.71c -9.19384081247e-09
      95243.71c -5.43644740853e-11
      94239.71c -0.000130467117992
      94238.71c -1.54018183284e-08
      63154.71c -4.40380799002e-08
      44103.71c -1.75548491863e-06
      92238.71c -0.062582562713
      54135.71c -3.17396271438e-08
      62149.71c -6.24311852574e-07
      54131.71c -2.10046753497e-05
      1001.71c -0.0171196088595
      62147.71c -3.27914283938e-06
      92234.71c -3.90621222388e-08
      92235.71c -0.0140995548237
      92236.71c -0.000255266047703
      43099.71c -3.42029222849e-05
      42095.71c -2.67607574948e-05
      61147.71c -1.45788149081e-05
      40090.71c -0.905506527356
      45103.71c -1.61070508959e-05
      55133.71c -5.00228438829e-05
      93237.71c -1.04391196575e-06
      55135.71c -3.46630975507e-05
      94240.71c -7.18044981509e-06
      94241.71c -4.65038393479e-07
      94242.71c -1.01345969037e-08
m5124 62152.71c -4.40152436848e-06
      62151.71c -1.86509098237e-06
      62150.71c -9.16181372509e-06
      64155.71c -6.2787645457e-09
      63153.71c -1.606249091e-06
      60143.71c -4.813615573e-05
      60145.71c -3.41678698432e-05
      95241.71c -1.07281524521e-08
      95243.71c -7.11334372186e-11
      94239.71c -0.0001366094347
      94238.71c -1.78373171624e-08
      63154.71c -4.8977286837e-08
      44103.71c -1.85038420968e-06
      92238.71c -0.0625775379069
      54135.71c -3.27888711669e-08
      62149.71c -6.22189076975e-07
      54131.71c -2.21540946989e-05
      1001.71c -0.0171206151341
      62147.71c -3.45833866079e-06
      92234.71c -4.11295891265e-08
      92235.71c -0.0140137119323
      92236.71c -0.000269146703169
      43099.71c -3.60877016618e-05
      42095.71c -2.82535568635e-05
      61147.71c -1.5355950715e-05
      40090.71c -0.905559752182
      45103.71c -1.69947317061e-05
      55133.71c -5.27770710685e-05
      93237.71c -1.14872033695e-06
      55135.71c -3.59277175511e-05
      94240.71c -7.94519034248e-06
      94241.71c -5.42017593025e-07
      94242.71c -1.25275648839e-08
c **  E-13  Fuel ID= 2987
m5131 62152.71c -8.38001202339e-07
      62151.71c -7.36874100074e-07
      62150.71c -1.84708881653e-06
      64155.71c -2.16462115377e-09
      63153.71c -3.73511216227e-07
      60143.71c -1.16494719685e-05
      60145.71c -8.63606461647e-06
      95241.71c -1.97442311651e-10
      95243.71c -2.73587863254e-13
      94239.71c -3.24213700532e-05
      94238.71c -6.58105860006e-10
      63154.71c -4.07788256282e-09
      44103.71c -1.06629165058e-06
      92238.71c -0.0626550727107
      54135.71c -2.39359838058e-08
      62149.71c -5.85189531297e-07
      54131.71c -5.4419415822e-06
      1001.71c -0.0171064656564
      62147.71c -4.79616966577e-07
      92234.71c -9.58001865068e-09
      92235.71c -0.0152498015466
      92236.71c -6.73849987692e-05
      43099.71c -9.00548205821e-06
      42095.71c -4.91536725108e-06
      61147.71c -4.1568195499e-06
      40090.71c -0.904811344639
      45103.71c -3.68878948908e-06
      55133.71c -1.30189777696e-05
      93237.71c -1.14112057573e-07
      55135.71c -1.02825705889e-05
      94240.71c -6.17200264494e-07
      94241.71c -1.49569408117e-08
      94242.71c -1.36237079073e-10
m5132 62152.71c -1.10860490534e-06
      62151.71c -8.77426441567e-07
      62150.71c -2.45453610728e-06
      64155.71c -2.33943408613e-09
      63153.71c -4.76467280958e-07
      60143.71c -1.46471008969e-05
      60145.71c -1.08760344005e-05
      95241.71c -5.19416713111e-10
      95243.71c -1.35476957181e-12
      94239.71c -4.66136212838e-05
      94238.71c -1.46221012694e-09
      63154.71c -6.8168758686e-09
      44103.71c -1.34617204333e-06
      92238.71c -0.0626435274503
      54135.71c -2.84944106508e-08
      62149.71c -6.14692058511e-07
      54131.71c -6.85327673522e-06
      1001.71c -0.0171077309167
      62147.71c -6.02515700564e-07
      92234.71c -1.34656029658e-08
      92235.71c -0.0151403262595
      92236.71c -8.6170328065e-05
      43099.71c -1.13468473422e-05
      42095.71c -6.19721498631e-06
      61147.71c -5.21933290128e-06
      40090.71c -0.90487826798
      45103.71c -4.65150106969e-06
      55133.71c -1.64025936386e-05
      93237.71c -1.92191885904e-07
      55135.71c -1.22695823294e-05
      94240.71c -1.13447357743e-06
      94241.71c -3.93226470072e-08
      94242.71c -4.57907362498e-10
m5133 62152.71c -1.38136535364e-06
      62151.71c -9.93331416924e-07
      62150.71c -3.04552480772e-06
      64155.71c -2.43448864526e-09
      63153.71c -5.74387171094e-07
      60143.71c -1.74719746245e-05
      60145.71c -1.29954987197e-05
      95241.71c -8.81084355096e-10
      95243.71c -3.3462146591e-12
      94239.71c -5.52251633459e-05
      94238.71c -2.33880334177e-09
      63154.71c -9.70539866663e-09
      44103.71c -1.61057552297e-06
      92238.71c -0.0626375179056
      54135.71c -3.22574149481e-08
      62149.71c -6.26723759551e-07
      54131.71c -8.18684447267e-06
      1001.71c -0.0171089348436
      62147.71c -7.18826082103e-07
      92234.71c -1.5992382348e-08
      92235.71c -0.0150375649185
      92236.71c -0.000102933981419
      43099.71c -1.35627173095e-05
      42095.71c -7.41195587461e-06
      61147.71c -6.22240591371e-06
      40090.71c -0.90494194722
      45103.71c -5.55939564465e-06
      55133.71c -1.96040557855e-05
      93237.71c -2.59173608357e-07
      55135.71c -1.39138264906e-05
      94240.71c -1.60618685256e-06
      94241.71c -6.66501725587e-08
      94242.71c -9.34744143883e-10
m5134 62152.71c -1.4788388734e-06
      62151.71c -1.03026222132e-06
      62150.71c -3.25147311989e-06
      64155.71c -2.45715976266e-09
      63153.71c -6.08723860001e-07
      60143.71c -1.84469868714e-05
      60145.71c -1.37281444094e-05
      95241.71c -1.03237955219e-09
      95243.71c -4.40039004737e-12
      94239.71c -5.81162545942e-05
      94238.71c -2.69411439858e-09
      63154.71c -1.08170423348e-08
      44103.71c -1.70171327244e-06
      92238.71c -0.0626354480683
      54135.71c -3.34550943014e-08
      62149.71c -6.29608561959e-07
      54131.71c -8.64775696729e-06
      1001.71c -0.0171093528937
      62147.71c -7.59037253133e-07
      92234.71c -1.69071258885e-08
      92235.71c -0.0150020558576
      92236.71c -0.000108710660308
      43099.71c -1.43294354887e-05
      42095.71c -7.83314764921e-06
      61147.71c -6.56864183762e-06
      40090.71c -0.90496405912
      45103.71c -5.87379748604e-06
      55133.71c -2.071160451e-05
      93237.71c -2.83092298736e-07
      55135.71c -1.44427078828e-05
      94240.71c -1.78558587655e-06
      94241.71c -7.80592685792e-08
      94242.71c -1.16013661168e-09
c **  E-14  Fuel ID= 3118
m5141 62152.71c -3.22127549049e-06
      62151.71c -1.53555561157e-06
      62150.71c -6.80204341766e-06
      64155.71c -6.94796369335e-09
      63153.71c -1.22280986454e-06
      60143.71c -3.67798141765e-05
      60145.71c -2.60125160226e-05
      95241.71c -2.31975541343e-08
      95243.71c -2.21316306465e-10
      94239.71c -8.95577619042e-05
      94238.71c -1.67848689176e-08
      63154.71c -3.44452673585e-08
      44103.71c -1.17975110078e-06
      92238.71c -0.0626175191929
      54135.71c -2.39724625465e-08
      62149.71c -6.18931869085e-07
      54131.71c -1.69083008214e-05
      1001.71c -0.0171159963408
      62147.71c -3.64587312512e-06
      92234.71c -2.78730882638e-08
      92235.71c -0.0144105403406
      92236.71c -0.000201616866778
      43099.71c -2.74957146251e-05
      42095.71c -2.25524435751e-05
      61147.71c -1.07975351238e-05
      40090.71c -0.905315450604
      45103.71c -1.31100578173e-05
      55133.71c -4.02643148273e-05
      93237.71c -7.6423391723e-07
      55135.71c -3.08000017609e-05
      94240.71c -5.04398622442e-06
      94241.71c -4.10488565076e-07
      94242.71c -1.98023298309e-08
m5142 62152.71c -4.16512851131e-06
      62151.71c -1.71486190646e-06
      62150.71c -8.59020352754e-06
      64155.71c -6.93988266065e-09
      63153.71c -1.57710448904e-06
      60143.71c -4.52222981559e-05
      60145.71c -3.21544450922e-05
      95241.71c -5.24157096682e-08
      95243.71c -9.28397604146e-10
      94239.71c -0.000124152468435
      94238.71c -3.84620930142e-08
      63154.71c -5.58099151526e-08
      44103.71c -1.46937272259e-06
      92238.71c -0.0625826239848
      54135.71c -2.8029424403e-08
      62149.71c -6.22112978611e-07
      54131.71c -2.08796908604e-05
      1001.71c -0.0171194835263
      62147.71c -4.46008240335e-06
      92234.71c -3.85938692334e-08
      92235.71c -0.0141119257411
      92236.71c -0.000252541800802
      43099.71c -3.40266804581e-05
      42095.71c -2.79230102389e-05
      61147.71c -1.32622650964e-05
      40090.71c -0.905499898115
      45103.71c -1.6225048863e-05
      55133.71c -4.98142814351e-05
      93237.71c -1.31730237531e-06
      55135.71c -3.61645363778e-05
      94240.71c -8.56636459513e-06
      94241.71c -9.40971029282e-07
      94242.71c -5.74228164642e-08
m5143 62152.71c -5.11174630512e-06
      62151.71c -1.83838414111e-06
      62150.71c -1.03107311103e-05
      64155.71c -6.79934117685e-09
      63153.71c -1.91927813249e-06
      60143.71c -5.31813803251e-05
      60145.71c -3.80201355721e-05
      95241.71c -8.00914054353e-08
      95243.71c -2.05849088778e-09
      94239.71c -0.000143502484703
      94238.71c -6.24658144907e-08
      63154.71c -7.77466448872e-08
      44103.71c -1.7410921143e-06
      92238.71c -0.0625642560421
      54135.71c -3.1276659758e-08
      62149.71c -6.14873121414e-07
      54131.71c -2.4662775023e-05
      1001.71c -0.0171228330777
      62147.71c -5.23656146265e-06
      92234.71c -4.51674581823e-08
      92235.71c -0.0138285514872
      92236.71c -0.000298256603661
      43099.71c -4.02697231941e-05
      42095.71c -3.30742972375e-05
      61147.71c -1.56120658999e-05
      40090.71c -0.90567706575
      45103.71c -1.91776076662e-05
      55133.71c -5.89395450607e-05
      93237.71c -1.79310300282e-06
      55135.71c -4.05954330627e-05
      94240.71c -1.15658843602e-05
      94241.71c -1.45740096435e-06
      94242.71c -1.06931367943e-07
m5144 62152.71c -5.43973815185e-06
      62151.71c -1.87337026904e-06
      62150.71c -1.08986599041e-05
      64155.71c -6.73911774602e-09
      63153.71c -2.03980311901e-06
      60143.71c -5.58634489798e-05
      60145.71c -4.00144331526e-05
      95241.71c -9.0567035352e-08
      95243.71c -2.61247532343e-09
      94239.71c -0.000149672016694
      94238.71c -7.24103120332e-08
      63154.71c -8.5985759378e-08
      44103.71c -1.8349358892e-06
      92238.71c -0.0625581282609
      54135.71c -3.23027722352e-08
      62149.71c -6.12272327293e-07
      54131.71c -2.59470498953e-05
      1001.71c -0.0171239776995
      62147.71c -5.4979977566e-06
      92234.71c -4.73191891617e-08
      92235.71c -0.0137322627981
      92236.71c -0.000313739529019
      43099.71c -4.23943697978e-05
      42095.71c -3.48267180054e-05
      61147.71c -1.64056341513e-05
      40090.71c -0.905737608168
      45103.71c -2.0179215788e-05
      55133.71c -6.20444773674e-05
      93237.71c -1.97208010978e-06
      55135.71c -4.19873969654e-05
      94240.71c -1.26565781045e-05
      94241.71c -1.65673357455e-06
      94242.71c -1.28677891866e-07
c **  E-15  Fuel ID= 2934
m5151 62152.71c -9.84633838599e-07
      62151.71c -8.28773656982e-07
      62150.71c -2.20959588457e-06
      64155.71c -2.62346879797e-09
      63153.71c -4.31385550786e-07
      60143.71c -1.35746326457e-05
      60145.71c -9.93778291222e-06
      95241.71c -2.57911099245e-10
      95243.71c -3.56910429914e-13
      94239.71c -3.72991252023e-05
      94238.71c -8.42817367009e-10
      63154.71c -5.0986386374e-09
      44103.71c -1.08034206457e-06
      92238.71c -0.0626522418884
      54135.71c -2.38506566299e-08
      62149.71c -5.95176944555e-07
      54131.71c -6.30515777657e-06
      1001.71c -0.01710717406
      62147.71c -6.03788068038e-07
      92234.71c -1.10463688044e-08
      92235.71c -0.0151868591115
      92236.71c -7.75199867727e-05
      43099.71c -1.03874666814e-05
      42095.71c -6.11114056723e-06
      61147.71c -4.77863031402e-06
      40090.71c -0.90484881419
      45103.71c -4.39081851255e-06
      55133.71c -1.5057983751e-05
      93237.71c -1.3885282206e-07
      55135.71c -1.1838808236e-05
      94240.71c -7.73452780115e-07
      94241.71c -1.93172981218e-08
      94242.71c -1.77571595684e-10
m5152 62152.71c -1.30764003196e-06
      62151.71c -9.82186691573e-07
      62150.71c -2.92389925465e-06
      64155.71c -2.81257745591e-09
      63153.71c -5.51677823616e-07
      60143.71c -1.70847609468e-05
      60145.71c -1.25311182851e-05
      95241.71c -6.81844181171e-10
      95243.71c -1.79274847158e-12
      94239.71c -5.3483161577e-05
      94238.71c -1.87809098597e-09
      63154.71c -8.54618917177e-09
      44103.71c -1.36664346545e-06
      92238.71c -0.0626389519911
      54135.71c -2.84359101956e-08
      62149.71c -6.20191928966e-07
      54131.71c -7.94966810416e-06
      1001.71c -0.0171086369591
      62147.71c -7.59411501765e-07
      92234.71c -1.55249308819e-08
      92235.71c -0.0150601499964
      92236.71c -9.92418395647e-05
      43099.71c -1.31048760322e-05
      42095.71c -7.7146232859e-06
      61147.71c -6.0055147419e-06
      40090.71c -0.904926191234
      45103.71c -5.54320800954e-06
      55133.71c -1.89951631137e-05
      93237.71c -2.3507805198e-07
      55135.71c -1.41389920359e-05
      94240.71c -1.42064066824e-06
      94241.71c -5.10420234175e-08
      94242.71c -6.00712413788e-10
m5153 62152.71c -1.6339911211e-06
      62151.71c -1.10591977124e-06
      62150.71c -3.61470046861e-06
      64155.71c -2.90147593653e-09
      63153.71c -6.66264538191e-07
      60143.71c -2.03907669636e-05
      60145.71c -1.49834170629e-05
      95241.71c -1.15902907255e-09
      95243.71c -4.44013849395e-12
      94239.71c -6.34025107237e-05
      94238.71c -3.01764598462e-09
      63154.71c -1.2199034989e-08
      44103.71c -1.63597034881e-06
      92238.71c -0.062631919138
      54135.71c -3.21841160784e-08
      62149.71c -6.29076106862e-07
      54131.71c -9.50292208579e-06
      1001.71c -0.0171100267887
      62147.71c -9.06590971881e-07
      92234.71c -1.8394312337e-08
      92235.71c -0.0149413026605
      92236.71c -0.000118618534886
      43099.71c -1.56755803232e-05
      42095.71c -9.23488045009e-06
      61147.71c -7.16325753025e-06
      40090.71c -0.904999703413
      45103.71c -6.6304111845e-06
      55133.71c -2.2719325566e-05
      93237.71c -3.18591127989e-07
      55135.71c -1.60395841045e-05
      94240.71c -2.01791694604e-06
      94241.71c -8.66957029455e-08
      94242.71c -1.23193198575e-09
m5154 62152.71c -1.74934939822e-06
      62151.71c -1.14461304723e-06
      62150.71c -3.85265206201e-06
      64155.71c -2.92061358761e-09
      63153.71c -7.05932515505e-07
      60143.71c -2.15189841692e-05
      60145.71c -1.58218599367e-05
      95241.71c -1.36062033218e-09
      95243.71c -5.86022880985e-12
      94239.71c -6.67275889285e-05
      94238.71c -3.49765271128e-09
      63154.71c -1.35889891359e-08
      44103.71c -1.72794220906e-06
      92238.71c -0.0626295049519
      54135.71c -3.33574887595e-08
      62149.71c -6.30974057206e-07
      54131.71c -1.00342875693e-05
      1001.71c -0.0171105035351
      62147.71c -9.56844849228e-07
      92234.71c -1.9436473782e-08
      92235.71c -0.0149006953549
      92236.71c -0.000125226000816
      43099.71c -1.65549624039e-05
      42095.71c -9.75583835732e-06
      61147.71c -7.55755068414e-06
      40090.71c -0.905024919936
      45103.71c -7.00255743909e-06
      55133.71c -2.39933997484e-05
      93237.71c -3.50843500723e-07
      55135.71c -1.66430572267e-05
      94240.71c -2.24356527814e-06
      94241.71c -1.01719226441e-07
      94242.71c -1.53130207712e-09
c **  E-16  Fuel ID= 4343
m5161 62152.71c -5.37474023693e-06
      62151.71c -1.94417989718e-06
      62150.71c -1.08285036482e-05
      64155.71c -1.00647054965e-08
      63153.71c -1.89650333353e-06
      60143.71c -5.69267752582e-05
      60145.71c -3.99416026881e-05
      95241.71c -2.17592311596e-08
      95243.71c -9.48658120237e-11
      94239.71c -0.000135825889715
      94238.71c -2.24885519065e-08
      63154.71c -6.08267359674e-08
      44103.71c -1.21159063514e-06
      92238.71c -0.0625858995927
      54135.71c -2.40739897073e-08
      62149.71c -5.9843351841e-07
      54131.71c -2.61050267649e-05
      1001.71c -0.0171236182829
      62147.71c -6.71257899608e-06
      92234.71c -4.29353882782e-08
      92235.71c -0.0137380764601
      92236.71c -0.000309447654904
      43099.71c -4.23210618544e-05
      42095.71c -3.76202403568e-05
      61147.71c -1.55658853911e-05
      40090.71c -0.905718597566
      45103.71c -2.0736793944e-05
      55133.71c -6.21093362324e-05
      93237.71c -1.3068003577e-06
      55135.71c -4.7338097062e-05
      94240.71c -9.17563819856e-06
      94241.71c -6.2174443469e-07
      94242.71c -1.67774552383e-08
m5162 62152.71c -7.01338693059e-06
      62151.71c -2.07730236806e-06
      62150.71c -1.36680504643e-05
      64155.71c -9.47154200038e-09
      63153.71c -2.4775605591e-06
      60143.71c -7.02591724708e-05
      60145.71c -4.960279143e-05
      95241.71c -5.36580030814e-08
      95243.71c -4.39101328219e-10
      94239.71c -0.00018655893875
      94238.71c -5.23110488182e-08
      63154.71c -9.98571523689e-08
      44103.71c -1.5105277809e-06
      92238.71c -0.0625313002695
      54135.71c -2.79255459901e-08
      62149.71c -5.96902682824e-07
      54131.71c -3.23856436211e-05
      1001.71c -0.0171291091199
      62147.71c -8.29576350424e-06
      92234.71c -5.92925829462e-08
      92235.71c -0.013269476848
      92236.71c -0.000389099628952
      43099.71c -5.26321773181e-05
      42095.71c -4.68595759242e-05
      61147.71c -1.91276335929e-05
      40090.71c -0.906009024102
      45103.71c -2.58027687702e-05
      55133.71c -7.72187863974e-05
      93237.71c -2.30029645494e-06
      55135.71c -5.5756574384e-05
      94240.71c -1.59660370339e-05
      94241.71c -1.52415918739e-06
      94242.71c -5.30265622738e-08
m5163 62152.71c -8.5970241432e-06
      62151.71c -2.13491915456e-06
      62150.71c -1.63287516888e-05
      64155.71c -8.89322191692e-09
      63153.71c -3.03348710238e-06
      60143.71c -8.24658054458e-05
      60145.71c -5.85922146537e-05
      95241.71c -8.70885927702e-08
      95243.71c -1.03713690472e-09
      94239.71c -0.000213383737332
      94238.71c -8.55391065068e-08
      63154.71c -1.39980238861e-07
      44103.71c -1.7787845892e-06
      92238.71c -0.0625022772852
      54135.71c -3.07890574245e-08
      62149.71c -5.85514614414e-07
      54131.71c -3.82134811666e-05
      1001.71c -0.0171342550708
      62147.71c -9.77871652553e-06
      92234.71c -6.89515205356e-08
      92235.71c -0.0128361007474
      92236.71c -0.000458860477213
      43099.71c -6.2238601575e-05
      42095.71c -5.55108513228e-05
      61147.71c -2.24140915329e-05
      40090.71c -0.906281208602
      45103.71c -3.04823018644e-05
      55133.71c -9.12914342936e-05
      93237.71c -3.16069606094e-06
      55135.71c -6.25010148265e-05
      94240.71c -2.18158052998e-05
      94241.71c -2.46405098822e-06
      94242.71c -1.04254221778e-07
m5164 62152.71c -9.14134506142e-06
      62151.71c -2.14703190852e-06
      62150.71c -1.72391890904e-05
      64155.71c -8.71367137768e-09
      63153.71c -3.23082545879e-06
      60143.71c -8.65748963256e-05
      60145.71c -6.16492650974e-05
      95241.71c -1.00519624112e-07
      95243.71c -1.33962672912e-09
      94239.71c -0.000221785059823
      94238.71c -9.93895115135e-08
      63154.71c -1.55129399508e-07
      44103.71c -1.86899115454e-06
      92238.71c -0.0624925772261
      54135.71c -3.16490114677e-08
      62149.71c -5.81508451969e-07
      54131.71c -4.01912951617e-05
      1001.71c -0.0171360154447
      62147.71c -1.02833378311e-05
      92234.71c -7.22132713178e-08
      92235.71c -0.0126888080469
      92236.71c -0.000482458456024
      43099.71c -6.55104964996e-05
      42095.71c -5.84684579692e-05
      61147.71c -2.3517516226e-05
      40090.71c -0.906374319962
      45103.71c -3.20733534056e-05
      55133.71c -9.60814622671e-05
      93237.71c -3.4811227077e-06
      55135.71c -6.4615581464e-05
      94240.71c -2.39425775905e-05
      94241.71c -2.84091098232e-06
      94242.71c -1.2768536173e-07
c **  E-17  Fuel ID= 3517
m5171 62152.71c -1.52350488879e-06
      62151.71c -1.17425681964e-06
      62150.71c -3.5500437219e-06
      64155.71c -4.25878743613e-09
      63153.71c -6.45491784526e-07
      60143.71c -2.06876748423e-05
      60145.71c -1.47666317812e-05
      95241.71c -4.45544727465e-10
      95243.71c -6.08714349877e-13
      94239.71c -5.4883386009e-05
      94238.71c -1.4659788731e-09
      63154.71c -8.74039407865e-09
      44103.71c -1.16039465428e-06
      92238.71c -0.0626423381384
      54135.71c -2.41369256307e-08
      62149.71c -6.35173406161e-07
      54131.71c -9.49923210897e-06
      1001.71c -0.0171098075257
      62147.71c -1.03838431134e-06
      92234.71c -1.62383422787e-08
      92235.71c -0.0149533766455
      92236.71c -0.000114998218091
      43099.71c -1.55089065676e-05
      42095.71c -1.04188215071e-05
      61147.71c -7.10358721461e-06
      40090.71c -0.904988105948
      45103.71c -6.96676978306e-06
      55133.71c -2.2606285459e-05
      93237.71c -2.253341801e-07
      55135.71c -1.75711797289e-05
      94240.71c -1.31902360675e-06
      94241.71c -3.38455904243e-08
      94242.71c -3.09704952625e-10
m5172 62152.71c -2.01821898342e-06
      62151.71c -1.36927958505e-06
      62150.71c -4.61703401033e-06
      64155.71c -4.50214702541e-09
      63153.71c -8.22241737184e-07
      60143.71c -2.58859241588e-05
      60145.71c -1.85129901942e-05
      95241.71c -1.16207002575e-09
      95243.71c -2.97152640263e-12
      94239.71c -7.81338456645e-05
      94238.71c -3.24549385426e-09
      63154.71c -1.44959837819e-08
      44103.71c -1.45754016427e-06
      92238.71c -0.0626229033786
      54135.71c -2.85244280988e-08
      62149.71c -6.42025967997e-07
      54131.71c -1.19083846484e-05
      1001.71c -0.0171119100917
      62147.71c -1.29997023448e-06
      92234.71c -2.28947189022e-08
      92235.71c -0.014770501276
      92236.71c -0.000146344903139
      43099.71c -1.9455265778e-05
      42095.71c -1.30892652917e-05
      61147.71c -8.87165280888e-06
      40090.71c -0.905099316857
      45103.71c -8.74872489262e-06
      55133.71c -2.83562424697e-05
      93237.71c -3.83382565382e-07
      55135.71c -2.08887071096e-05
      94240.71c -2.39881475835e-06
      94241.71c -8.81277364045e-08
      94242.71c -1.02703954811e-09
m5173 62152.71c -2.52249608418e-06
      62151.71c -1.52059785907e-06
      62150.71c -5.64362637671e-06
      64155.71c -4.58512766816e-09
      63153.71c -9.9182911091e-07
      60143.71c -3.08090452367e-05
      60145.71c -2.20778072074e-05
      95241.71c -1.96995799348e-09
      95243.71c -7.31254990575e-12
      94239.71c -9.23481848612e-05
      94238.71c -5.17513990406e-09
      63154.71c -2.06000174637e-08
      44103.71c -1.73837379368e-06
      92238.71c -0.0626125442288
      54135.71c -3.21215337164e-08
      62149.71c -6.38178121885e-07
      54131.71c -1.41979396441e-05
      1001.71c -0.0171139210847
      62147.71c -1.54902927337e-06
      92234.71c -2.6907293712e-08
      92235.71c -0.0145978780625
      92236.71c -0.000174460414356
      43099.71c -2.32120056728e-05
      42095.71c -1.56372725687e-05
      61147.71c -1.05502835823e-05
      40090.71c -0.905205684198
      45103.71c -1.0439924744e-05
      55133.71c -3.38289766281e-05
      93237.71c -5.19258725401e-07
      55135.71c -2.36470864628e-05
      94240.71c -3.39747017428e-06
      94241.71c -1.4916412234e-07
      94242.71c -2.09495284598e-09
m5174 62152.71c -2.70373939683e-06
      62151.71c -1.5670724339e-06
      62150.71c -6.00095190635e-06
      64155.71c -4.5947890476e-09
      63153.71c -1.05162373578e-06
      60143.71c -3.25138328534e-05
      60145.71c -2.33155844238e-05
      95241.71c -2.31148306624e-09
      95243.71c -9.56766651725e-12
      94239.71c -9.69885104335e-05
      94238.71c -5.97322218051e-09
      63154.71c -2.29536216215e-08
      44103.71c -1.83531453291e-06
      92238.71c -0.0626091752883
      54135.71c -3.32683327777e-08
      62149.71c -6.36768709124e-07
      54131.71c -1.49929275181e-05
      1001.71c -0.017114620823
      62147.71c -1.63570766102e-06
      92234.71c -2.83543752263e-08
      92235.71c -0.0145380096758
      92236.71c -0.000184171038647
      43099.71c -2.45171356047e-05
      42095.71c -1.65260710517e-05
      61147.71c -1.11307486527e-05
      40090.71c -0.905242695415
      45103.71c -1.10280019582e-05
      55133.71c -3.57299324134e-05
      93237.71c -5.69352453056e-07
      55135.71c -2.45364386922e-05
      94240.71c -3.77317324477e-06
      94241.71c -1.74803578964e-07
      94242.71c -2.60299261121e-09
c **  E-18  Fuel ID= 3502
m5181 62152.71c -1.54650942792e-06
      62151.71c -1.1835563109e-06
      62150.71c -3.60093090974e-06
      64155.71c -4.29089635266e-09
      63153.71c -6.53332103132e-07
      60143.71c -2.09383661797e-05
      60145.71c -1.49428817261e-05
      95241.71c -4.58157761131e-10
      95243.71c -6.33173540272e-13
      94239.71c -5.50728245227e-05
      94238.71c -1.4883641379e-09
      63154.71c -8.91920035369e-09
      44103.71c -1.16909838663e-06
      92238.71c -0.0626424853241
      54135.71c -2.42415713204e-08
      62149.71c -6.3453295882e-07
      54131.71c -9.61425605551e-06
      1001.71c -0.017109903925
      62147.71c -1.05696370808e-06
      92234.71c -1.62467518782e-08
      92235.71c -0.0149449219588
      92236.71c -0.000116275072784
      43099.71c -1.5695338156e-05
      42095.71c -1.0570705612e-05
      61147.71c -7.18349635688e-06
      40090.71c -0.904993204791
      45103.71c -7.05458531623e-06
      55133.71c -2.28798468186e-05
      93237.71c -2.27498609612e-07
      55135.71c -1.77541247665e-05
      94240.71c -1.33949635106e-06
      94241.71c -3.4617249244e-08
      94242.71c -3.20686723781e-10
m5182 62152.71c -2.03208628537e-06
      62151.71c -1.37344613746e-06
      62150.71c -4.64584683933e-06
      64155.71c -4.51966806286e-09
      63153.71c -8.26662151078e-07
      60143.71c -2.60266185267e-05
      60145.71c -1.86113673129e-05
      95241.71c -1.17280744549e-09
      95243.71c -3.00815520796e-12
      94239.71c -7.81838378906e-05
      94238.71c -3.25588214821e-09
      63154.71c -1.46133638659e-08
      44103.71c -1.46093893432e-06
      92238.71c -0.062622969316
      54135.71c -2.85318427937e-08
      62149.71c -6.4135585374e-07
      54131.71c -1.19731373612e-05
      1001.71c -0.0171119647385
      62147.71c -1.31397324999e-06
      92234.71c -2.2755634471e-08
      92235.71c -0.0147658236476
      92236.71c -0.000147016356276
      43099.71c -1.95599116062e-05
      42095.71c -1.3186965296e-05
      61147.71c -8.91321464168e-06
      40090.71c -0.905102207287
      45103.71c -8.79909186126e-06
      55133.71c -2.85105401891e-05
      93237.71c -3.83866804273e-07
      55135.71c -2.10003820587e-05
      94240.71c -2.41100262736e-06
      94241.71c -8.85189972519e-08
      94242.71c -1.03794303103e-09
m5183 62152.71c -2.53880026264e-06
      62151.71c -1.52428921368e-06
      62150.71c -5.67528902903e-06
      64155.71c -4.60014930171e-09
      63153.71c -9.96738735455e-07
      60143.71c -3.09658681929e-05
      60145.71c -2.21864845108e-05
      95241.71c -1.98629053567e-09
      95243.71c -7.36640414086e-12
      94239.71c -9.2283444143e-05
      94238.71c -5.19366595897e-09
      63154.71c -2.07414022097e-08
      44103.71c -1.74057105147e-06
      92238.71c -0.0626127743925
      54135.71c -3.20778629828e-08
      62149.71c -6.37590512523e-07
      54131.71c -1.42702065811e-05
      1001.71c -0.0171139805746
      62147.71c -1.56528756251e-06
      92234.71c -2.68555276693e-08
      92235.71c -0.0145927389092
      92236.71c -0.000175179190845
      43099.71c -2.33278837794e-05
      42095.71c -1.57499178802e-05
      61147.71c -1.05961173294e-05
      40090.71c -0.905208830795
      45103.71c -1.04968481539e-05
      55133.71c -3.40004371707e-05
      93237.71c -5.19813942804e-07
      55135.71c -2.3767458003e-05
      94240.71c -3.40998127924e-06
      94241.71c -1.49534867743e-07
      94242.71c -2.11329635782e-09
m5184 62152.71c -2.71781382631e-06
      62151.71c -1.57000824238e-06
      62150.71c -6.02778947957e-06
      64155.71c -4.60894727382e-09
      63153.71c -1.05570380799e-06
      60143.71c -3.264684192e-05
      60145.71c -2.34073611865e-05
      95241.71c -2.32988382929e-09
      95243.71c -9.64361345038e-12
      94239.71c -9.70113875036e-05
      94238.71c -5.99155416652e-09
      63154.71c -2.30723852173e-08
      44103.71c -1.83647905707e-06
      92238.71c -0.0626093118299
      54135.71c -3.31850215324e-08
      62149.71c -6.36281046548e-07
      54131.71c -1.50540379545e-05
      1001.71c -0.0171146702961
      62147.71c -1.65123776145e-06
      92234.71c -2.82049698264e-08
      92235.71c -0.0145336637132
      92236.71c -0.000184786788706
      43099.71c -2.46154245957e-05
      42095.71c -1.66276929915e-05
      61147.71c -1.11678491566e-05
      40090.71c -0.905245312192
      45103.71c -1.10766578845e-05
      55133.71c -3.58758514887e-05
      93237.71c -5.70285447692e-07
      55135.71c -2.46455140764e-05
      94240.71c -3.78561192779e-06
      94241.71c -1.75325801649e-07
      94242.71c -2.62296078996e-09
c **  E-19  Fuel ID= 3503
m5191 62152.71c -2.46502754195e-06
      62151.71c -1.48778468284e-06
      62150.71c -5.52028412729e-06
      64155.71c -6.39261441166e-09
      63153.71c -9.64747960538e-07
      60143.71c -3.06527106739e-05
      60145.71c -2.16017428569e-05
      95241.71c -2.07707114364e-09
      95243.71c -4.48358183861e-12
      94239.71c -7.85034639447e-05
      94238.71c -4.10999999507e-09
      63154.71c -1.85454613679e-08
      44103.71c -1.17624730693e-06
      92238.71c -0.0626274999102
      54135.71c -2.40868094227e-08
      62149.71c -6.26298041288e-07
      54131.71c -1.40171219181e-05
      1001.71c -0.0171135332492
      62147.71c -2.18023752371e-06
      92234.71c -2.36088390597e-08
      92235.71c -0.0146231547311
      92236.71c -0.000167971954348
      43099.71c -2.27752216453e-05
      42095.71c -1.77273239256e-05
      61147.71c -9.83362921728e-06
      40090.71c -0.905185170435
      45103.71c -1.07084043636e-05
      55133.71c -3.33163445163e-05
      93237.71c -4.33926946592e-07
      55135.71c -2.56836943083e-05
      94240.71c -2.80885004044e-06
      94241.71c -1.06344633636e-07
      94242.71c -1.48852944634e-09
m5192 62152.71c -3.26610155841e-06
      62151.71c -1.68473309666e-06
      62150.71c -7.08083754892e-06
      64155.71c -6.42948038845e-09
      63153.71c -1.23761154986e-06
      60143.71c -3.82161727512e-05
      60145.71c -2.70158010718e-05
      95241.71c -5.33056942679e-09
      95243.71c -2.15665714701e-11
      94239.71c -0.000110686768945
      94238.71c -9.32157899134e-09
      63154.71c -3.06537404688e-08
      44103.71c -1.47613555333e-06
      92238.71c -0.0625985664325
      54135.71c -2.83595146818e-08
      62149.71c -6.30436030564e-07
      54131.71c -1.75246747961e-05
      1001.71c -0.0171165765576
      62147.71c -2.71925989128e-06
      92234.71c -3.30933624318e-08
      92235.71c -0.0143593823168
      92236.71c -0.000213084797282
      43099.71c -2.85066970975e-05
      42095.71c -2.2217434547e-05
      61147.71c -1.22285142965e-05
      40090.71c -0.905346139979
      45103.71c -1.34152594347e-05
      55133.71c -4.16945682762e-05
      93237.71c -7.52979483867e-07
      55135.71c -3.04593808699e-05
      94240.71c -5.04653900061e-06
      94241.71c -2.71930652245e-07
      94242.71c -4.87030866882e-09
m5193 62152.71c -4.07219416301e-06
      62151.71c -1.82055283966e-06
      62150.71c -8.56888340826e-06
      64155.71c -6.3002431743e-09
      63153.71c -1.49968564961e-06
      60143.71c -4.53163906511e-05
      60145.71c -3.2135631981e-05
      95241.71c -8.92611576053e-09
      95243.71c -5.23687108084e-11
      94239.71c -0.00012956069116
      94238.71c -1.5067804494e-08
      63154.71c -4.33670821326e-08
      44103.71c -1.75526476295e-06
      92238.71c -0.0625833146776
      54135.71c -3.17532437861e-08
      62149.71c -6.246563153e-07
      54131.71c -2.0836467225e-05
      1001.71c -0.017119467774
      62147.71c -3.23043115847e-06
      92234.71c -3.87640338648e-08
      92235.71c -0.0141118690244
      92236.71c -0.000253272216741
      43099.71c -3.39308082773e-05
      42095.71c -2.64830157349e-05
      61147.71c -1.44858254051e-05
      40090.71c -0.905499064929
      45103.71c -1.59667832045e-05
      55133.71c -4.96219410739e-05
      93237.71c -1.02899084622e-06
      55135.71c -3.43910557485e-05
      94240.71c -7.07329215959e-06
      94241.71c -4.54752745278e-07
      94242.71c -9.83259350881e-09
m5194 62152.71c -4.35792156245e-06
      62151.71c -1.85946221078e-06
      62150.71c -9.08357836896e-06
      64155.71c -6.24316683879e-09
      63153.71c -1.59204524369e-06
      60143.71c -4.77527373838e-05
      60145.71c -3.38999551609e-05
      95241.71c -1.04168269722e-08
      95243.71c -6.85325158867e-11
      94239.71c -0.000135669802665
      94238.71c -1.74496680518e-08
      63154.71c -4.82340795406e-08
      44103.71c -1.85029862726e-06
      92238.71c -0.0625783315143
      54135.71c -3.28054783954e-08
      62149.71c -6.22548145732e-07
      54131.71c -2.19774126721e-05
      1001.71c -0.0171204666681
      62147.71c -3.40705876744e-06
      92234.71c -4.08170995584e-08
      92235.71c -0.0140266469245
      92236.71c -0.000267053771358
      43099.71c -3.58016938291e-05
      42095.71c -2.79609766299e-05
      61147.71c -1.52586669345e-05
      40090.71c -0.905551899381
      45103.71c -1.68472263827e-05
      55133.71c -5.23556867187e-05
      93237.71c -1.13226869339e-06
      55135.71c -3.56467855496e-05
      94240.71c -7.82732289758e-06
      94241.71c -5.30102149188e-07
      94242.71c -1.2155570141e-08
c **  E-20  Fuel ID= 3009
m5201 62152.71c -3.06839034785e-06
      62151.71c -1.62557176024e-06
      62150.71c -6.68607003929e-06
      64155.71c -7.42768047219e-09
      63153.71c -1.16063672506e-06
      60143.71c -3.64926664292e-05
      60145.71c -2.56370516695e-05
      95241.71c -4.19227757394e-09
      95243.71c -1.13437573826e-11
      94239.71c -9.18189006612e-05
      94238.71c -6.6615444552e-09
      63154.71c -2.60239538408e-08
      44103.71c -1.1823221512e-06
      92238.71c -0.0626184944341
      54135.71c -2.40105020788e-08
      62149.71c -6.20537401642e-07
      54131.71c -1.66820076155e-05
      1001.71c -0.0171157424023
      62147.71c -3.0211993808e-06
      92234.71c -2.79219413098e-08
      92235.71c -0.0144283049727
      92236.71c -0.000199176484302
      43099.71c -2.70706611268e-05
      42095.71c -2.20925989945e-05
      61147.71c -1.12663532522e-05
      40090.71c -0.905302019049
      45103.71c -1.29190245964e-05
      55133.71c -3.96468980552e-05
      93237.71c -5.89494899164e-07
      55135.71c -3.04615397354e-05
      94240.71c -3.9431276878e-06
      94241.71c -1.78293040906e-07
      94242.71c -3.06240317308e-09
m5202 62152.71c -4.05543522682e-06
      62151.71c -1.81293776948e-06
      62150.71c -8.53276939527e-06
      64155.71c -7.31648607398e-09
      63153.71c -1.49522508954e-06
      60143.71c -4.53918281495e-05
      60145.71c -3.20100170314e-05
      95241.71c -1.06427529579e-08
      95243.71c -5.40427157089e-11
      94239.71c -0.000128649143024
      94238.71c -1.52471054142e-08
      63154.71c -4.29273953562e-08
      44103.71c -1.48145337112e-06
      92238.71c -0.0625840042023
      54135.71c -2.81766833602e-08
      62149.71c -6.23475442823e-07
      54131.71c -2.08192799504e-05
      1001.71c -0.0171193310589
      62147.71c -3.75976265479e-06
      92234.71c -3.89728450461e-08
      92235.71c -0.0141181652026
      92236.71c -0.000252147390915
      43099.71c -3.38325066479e-05
      42095.71c -2.76494808794e-05
      61147.71c -1.39719309311e-05
      40090.71c -0.905491833664
      45103.71c -1.61599756723e-05
      55133.71c -4.95413835617e-05
      93237.71c -1.02945835958e-06
      55135.71c -3.60685669686e-05
      94240.71c -7.0296037099e-06
      94241.71c -4.50986294196e-07
      94242.71c -9.92399880266e-09
m5203 62152.71c -5.03955816802e-06
      62151.71c -1.93259631848e-06
      62150.71c -1.0286349552e-05
      64155.71c -7.06360215111e-09
      63153.71c -1.81656741108e-06
      60143.71c -5.36998654138e-05
      60145.71c -3.80152417868e-05
      95241.71c -1.76786340936e-08
      95243.71c -1.30057627163e-10
      94239.71c -0.000149781986416
      94238.71c -2.47553559281e-08
      63154.71c -6.05819870196e-08
      44103.71c -1.75810570345e-06
      92238.71c -0.0625658071168
      54135.71c -3.14608870057e-08
      62149.71c -6.16472117707e-07
      54131.71c -2.47105144521e-05
      1001.71c -0.0171227295801
      62147.71c -4.45823209941e-06
      92234.71c -4.55984286183e-08
      92235.71c -0.013828084786
      92236.71c -0.000299155485376
      43099.71c -4.02095860734e-05
      42095.71c -3.29112633284e-05
      61147.71c -1.65105864529e-05
      40090.71c -0.905671591456
      45103.71c -1.92010168163e-05
      55133.71c -5.88705365769e-05
      93237.71c -1.41030904148e-06
      55135.71c -4.06571487212e-05
      94240.71c -9.79061154645e-06
      94241.71c -7.47865111917e-07
      94242.71c -1.98937018562e-08
m5204 62152.71c -5.38551255027e-06
      62151.71c -1.96541875903e-06
      62150.71c -1.08911089155e-05
      64155.71c -6.97029861604e-09
      63153.71c -1.9301203596e-06
      60143.71c -5.65373813713e-05
      60145.71c -4.00772276258e-05
      95241.71c -2.05676043792e-08
      95243.71c -1.69456318989e-10
      94239.71c -0.000156606744393
      94238.71c -2.87112845937e-08
      63154.71c -6.7316490227e-08
      44103.71c -1.85181724829e-06
      92238.71c -0.0625597454509
      54135.71c -3.24679957776e-08
      62149.71c -6.13956958936e-07
      54131.71c -2.60458211573e-05
      1001.71c -0.0171239016352
      62147.71c -4.69858472376e-06
      92234.71c -4.79400009199e-08
      92235.71c -0.0137285373496
      92236.71c -0.000315219577204
      43099.71c -4.24015892932e-05
      42095.71c -3.47276561077e-05
      61147.71c -1.73756322699e-05
      40090.71c -0.905733584906
      45103.71c -2.02463978862e-05
      55133.71c -6.20765038207e-05
      93237.71c -1.55356471804e-06
      55135.71c -4.21159467171e-05
      94240.71c -1.0812171945e-05
      94241.71c -8.69250618589e-07
      94242.71c -2.45301955393e-08
c **  E-21  Fuel ID= 3011
m5211 62152.71c -3.32406993363e-06
      62151.71c -1.67599309783e-06
      62150.71c -7.16789199847e-06
      64155.71c -7.82007638605e-09
      63153.71c -1.2427507239e-06
      60143.71c -3.88969761191e-05
      60145.71c -2.73037865138e-05
      95241.71c -5.28557474917e-09
      95243.71c -1.51547601951e-11
      94239.71c -9.72141137479e-05
      94238.71c -7.88734317714e-09
      63154.71c -2.93724269448e-08
      44103.71c -1.18467243217e-06
      92238.71c -0.0626147403336
      54135.71c -2.39927913371e-08
      62149.71c -6.1812496972e-07
      54131.71c -1.7782037857e-05
      1001.71c -0.0171166564983
      62147.71c -3.39291198035e-06
      92234.71c -2.9694620839e-08
      92235.71c -0.0143478427084
      92236.71c -0.000212054453953
      43099.71c -2.88455315846e-05
      42095.71c -2.38996831748e-05
      61147.71c -1.18315816883e-05
      40090.71c -0.905350368276
      45103.71c -1.38319753489e-05
      55133.71c -4.22623042319e-05
      93237.71c -6.58848067256e-07
      55135.71c -3.24331118604e-05
      94240.71c -4.45041664727e-06
      94241.71c -2.12997566712e-07
      94242.71c -3.88238055507e-09
m5212 62152.71c -4.38824056353e-06
      62151.71c -1.85823603387e-06
      62150.71c -9.13210951887e-06
      64155.71c -7.6465815066e-09
      63153.71c -1.60362878695e-06
      60143.71c -4.83382310756e-05
      60145.71c -3.40691495595e-05
      95241.71c -1.33761229973e-08
      95243.71c -7.20198091694e-11
      94239.71c -0.00013587645743
      94238.71c -1.81012850286e-08
      63154.71c -4.84150557609e-08
      44103.71c -1.48348112627e-06
      92238.71c -0.0625779617793
      54135.71c -2.81200585057e-08
      62149.71c -6.20533567979e-07
      54131.71c -2.21764399755e-05
      1001.71c -0.0171204689975
      62147.71c -4.21892098242e-06
      92234.71c -4.13836832741e-08
      92235.71c -0.014018748968
      92236.71c -0.000268232252632
      43099.71c -3.60295170964e-05
      42095.71c -2.98946298914e-05
      61147.71c -1.46569111257e-05
      40090.71c -0.905552022586
      45103.71c -1.72914403907e-05
      55133.71c -5.2777674247e-05
      93237.71c -1.15265174273e-06
      55135.71c -3.83791370217e-05
      94240.71c -7.91143460605e-06
      94241.71c -5.3692976333e-07
      94242.71c -1.25469377017e-08
m5213 62152.71c -5.44517373745e-06
      62151.71c -1.97059885623e-06
      62150.71c -1.0993914957e-05
      64155.71c -7.34613099181e-09
      63153.71c -1.95011831774e-06
      60143.71c -5.71319175575e-05
      60145.71c -4.04337915311e-05
      95241.71c -2.21659354575e-08
      95243.71c -1.72904600746e-10
      94239.71c -0.000157867240196
      94238.71c -2.94233088587e-08
      63154.71c -6.8263093509e-08
      44103.71c -1.75850597552e-06
      92238.71c -0.062558546789
      54135.71c -3.13480469883e-08
      62149.71c -6.13022233194e-07
      54131.71c -2.63026865252e-05
      1001.71c -0.0171240744079
      62147.71c -4.99928827884e-06
      92234.71c -4.83966015352e-08
      92235.71c -0.0137113945825
      92236.71c -0.00031799998088
      43099.71c -4.27940723438e-05
      42095.71c -3.55640127525e-05
      61147.71c -1.73024971663e-05
      40090.71c -0.905742723359
      45103.71c -2.05315056131e-05
      55133.71c -6.26768262814e-05
      93237.71c -1.57998778565e-06
      55135.71c -4.32326284724e-05
      94240.71c -1.09931999649e-05
      94241.71c -8.87678505751e-07
      94242.71c -2.50974711119e-08
m5214 62152.71c -5.81587669897e-06
      62151.71c -2.00081513276e-06
      62150.71c -1.16359269731e-05
      64155.71c -7.23722490603e-09
      63153.71c -2.07282980813e-06
      60143.71c -6.01319104968e-05
      60145.71c -4.26183484579e-05
      95241.71c -2.57651959437e-08
      95243.71c -2.25052214946e-10
      94239.71c -0.000164969215375
      94238.71c -3.41427460711e-08
      63154.71c -7.58354677695e-08
      44103.71c -1.85228566273e-06
      92238.71c -0.0625520007774
      54135.71c -3.23447839177e-08
      62149.71c -6.10315512887e-07
      54131.71c -2.77175961023e-05
      1001.71c -0.0171253187237
      62147.71c -5.26740567972e-06
      92234.71c -5.08418877509e-08
      92235.71c -0.0136059501198
      92236.71c -0.000335002200155
      43099.71c -4.51184370849e-05
      42095.71c -3.7518354087e-05
      61147.71c -1.82031321792e-05
      40090.71c -0.90580853889
      45103.71c -2.16440698962e-05
      55133.71c -6.60770431465e-05
      93237.71c -1.74130086746e-06
      55135.71c -4.47743045766e-05
      94240.71c -1.21318932561e-05
      94241.71c -1.03090941019e-06
      94242.71c -3.09262200671e-08
c **  E-22  Fuel ID= 3014
m5221 62152.71c -3.28389370643e-06
      62151.71c -1.66807013789e-06
      62150.71c -7.09218087573e-06
      64155.71c -7.75841723035e-09
      63153.71c -1.22984773572e-06
      60143.71c -3.85191747703e-05
      60145.71c -2.7041884033e-05
      95241.71c -5.1137794313e-09
      95243.71c -1.45559180372e-11
      94239.71c -9.63663366127e-05
      94238.71c -7.69472722071e-09
      63154.71c -2.88462644208e-08
      44103.71c -1.18430312062e-06
      92238.71c -0.0626153302344
      54135.71c -2.39955743156e-08
      62149.71c -6.18504047356e-07
      54131.71c -1.76091845383e-05
      1001.71c -0.0171165128617
      62147.71c -3.3345028985e-06
      92234.71c -2.94160707338e-08
      92235.71c -0.0143604861512
      92236.71c -0.000210030873412
      43099.71c -2.85666372077e-05
      42095.71c -2.36157268926e-05
      61147.71c -1.17427644838e-05
      40090.71c -0.905342770917
      45103.71c -1.36885187771e-05
      55133.71c -4.18513322239e-05
      93237.71c -6.47950252986e-07
      55135.71c -3.21233087551e-05
      94240.71c -4.37070376589e-06
      94241.71c -2.07544268947e-07
      94242.71c -3.75353335947e-09
m5222 62152.71c -4.33594517678e-06
      62151.71c -1.85111808835e-06
      62150.71c -9.03793217224e-06
      64155.71c -7.5947119408e-09
      63153.71c -1.58659476539e-06
      60143.71c -4.78752478733e-05
      60145.71c -3.3745587645e-05
      95241.71c -1.29466147306e-08
      95243.71c -6.91949774994e-11
      94239.71c -0.000134740792943
      94238.71c -1.7652793342e-08
      63154.71c -4.75527519107e-08
      44103.71c -1.48316249484e-06
      92238.71c -0.0625789112558
      54135.71c -2.81289562557e-08
      62149.71c -6.20995839661e-07
      54131.71c -2.19631825527e-05
      1001.71c -0.0171202901874
      62147.71c -4.14677111079e-06
      92234.71c -4.10048560631e-08
      92235.71c -0.014034370744
      92236.71c -0.000265704756912
      43099.71c -3.56842897265e-05
      42095.71c -2.95418382654e-05
      61147.71c -1.45492767213e-05
      40090.71c -0.905542564796
      45103.71c -1.71136476136e-05
      55133.71c -5.22691395141e-05
      93237.71c -1.13329374301e-06
      55135.71c -3.80160654564e-05
      94240.71c -7.77286805476e-06
      94241.71c -5.23425031101e-07
      94242.71c -1.21347820389e-08
m5223 62152.71c -5.38143731038e-06
      62151.71c -1.96462732516e-06
      62150.71c -1.08827316242e-05
      64155.71c -7.30173580531e-09
      63153.71c -1.92913278793e-06
      60143.71c -5.6592621836e-05
      60145.71c -4.00537525705e-05
      95241.71c -2.14608230859e-08
      95243.71c -1.66171838957e-10
      94239.71c -0.000156596763353
      94238.71c -2.86898097922e-08
      63154.71c -6.70561223754e-08
      44103.71c -1.75844307874e-06
      92238.71c -0.0625596876411
      54135.71c -3.13657781113e-08
      62149.71c -6.13564331006e-07
      54131.71c -2.60525004701e-05
      1001.71c -0.0171238630883
      62147.71c -4.91426938331e-06
      92234.71c -4.79569104644e-08
      92235.71c -0.0137297307047
      92236.71c -0.000315038849937
      43099.71c -4.23879589371e-05
      42095.71c -3.51471728111e-05
      61147.71c -1.71780602284e-05
      40090.71c -0.905731546043
      45103.71c -2.03224391754e-05
      55133.71c -6.20787247181e-05
      93237.71c -1.55332530591e-06
      55135.71c -4.28279303056e-05
      94240.71c -1.08042311609e-05
      94241.71c -8.6570891964e-07
      94242.71c -2.42797765147e-08
m5224 62152.71c -5.74825140239e-06
      62151.71c -1.99525312143e-06
      62150.71c -1.15188899425e-05
      64155.71c -7.19528142728e-09
      63153.71c -2.05040515029e-06
      60143.71c -5.95670839449e-05
      60145.71c -4.2219049282e-05
      95241.71c -2.49484720783e-08
      95243.71c -2.16316150467e-10
      94239.71c -0.00016365517799
      94238.71c -3.32892730428e-08
      63154.71c -7.44968377173e-08
      44103.71c -1.8522120584e-06
      92238.71c -0.0625532177372
      54135.71c -3.23641448207e-08
      62149.71c -6.10887711732e-07
      54131.71c -2.7454901645e-05
      1001.71c -0.0171250960495
      62147.71c -5.17802396434e-06
      92234.71c -5.03858995879e-08
      92235.71c -0.0136252128716
      92236.71c -0.000331893656499
      43099.71c -4.46915250426e-05
      42095.71c -3.70798375909e-05
      61147.71c -1.80731029308e-05
      40090.71c -0.905796760991
      45103.71c -2.14244466203e-05
      55133.71c -6.54484181572e-05
      93237.71c -1.71180093625e-06
      55135.71c -4.43565833533e-05
      94240.71c -1.19245187679e-05
      94241.71c -1.00550714618e-06
      94242.71c -2.99211803654e-08
c **  E-23  Fuel ID= 3008
m5231 62152.71c -2.79550865677e-06
      62151.71c -1.50645673185e-06
      62150.71c -6.07742633774e-06
      64155.71c -6.60437716358e-09
      63153.71c -1.07779281314e-06
      60143.71c -3.33061200907e-05
      60145.71c -2.35183414229e-05
      95241.71c -1.17585701266e-08
      95243.71c -1.04665354236e-10
      94239.71c -8.28865759279e-05
      94238.71c -9.85386445482e-09
      63154.71c -2.56100761205e-08
      44103.71c -1.18075880326e-06
      92238.71c -0.0626235030611
      54135.71c -2.40611083545e-08
      62149.71c -6.22894089761e-07
      54131.71c -1.52722800819e-05
      1001.71c -0.017114606208
      62147.71c -2.82319401058e-06
      92234.71c -2.53188762198e-08
      92235.71c -0.0145308179069
      92236.71c -0.000182510293332
      43099.71c -2.48266069671e-05
      42095.71c -1.98078678889e-05
      61147.71c -1.02446757201e-05
      40090.71c -0.905241922388
      45103.71c -1.1748113884e-05
      55133.71c -3.63350476027e-05
      93237.71c -5.79279965187e-07
      55135.71c -2.78837192631e-05
      94240.71c -3.79061658711e-06
      94241.71c -2.43653091723e-07
      94242.71c -9.9011520198e-09
m5232 62152.71c -3.6428350966e-06
      62151.71c -1.69297736656e-06
      62150.71c -7.70805713728e-06
      64155.71c -6.62558224888e-09
      63153.71c -1.38173627699e-06
      60143.71c -4.11082455798e-05
      60145.71c -2.91490826412e-05
      95241.71c -2.68394684624e-08
      95243.71c -4.40089559671e-10
      94239.71c -0.000115789934469
      94238.71c -2.24560240948e-08
      63154.71c -4.16221001412e-08
      44103.71c -1.47225093355e-06
      92238.71c -0.0625922325439
      54135.71c -2.81828668689e-08
      62149.71c -6.26819260873e-07
      54131.71c -1.89161435854e-05
      1001.71c -0.0171177860152
      62147.71c -3.47186584641e-06
      92234.71c -3.51760626916e-08
      92235.71c -0.0142567486821
      92236.71c -0.000229353286894
      43099.71c -3.08002469742e-05
      42095.71c -2.45913801429e-05
      61147.71c -1.26246479218e-05
      40090.71c -0.905410111752
      45103.71c -1.45811271151e-05
      55133.71c -4.50681005347e-05
      93237.71c -9.9685718477e-07
      55135.71c -3.28252382113e-05
      94240.71c -6.55915048054e-06
      94241.71c -5.70750642321e-07
      94242.71c -2.89302863612e-08
m5233 62152.71c -4.50375517101e-06
      62151.71c -1.82375358961e-06
      62150.71c -9.28787418043e-06
      64155.71c -6.49913135583e-09
      63153.71c -1.67718786348e-06
      60143.71c -4.85341029067e-05
      60145.71c -3.45611838347e-05
      95241.71c -4.13812427093e-08
      95243.71c -9.77768836298e-10
      94239.71c -0.000134696869027
      94238.71c -3.64273036641e-08
      63154.71c -5.83038220473e-08
      44103.71c -1.74706765149e-06
      92238.71c -0.0625758181477
      54135.71c -3.1505586774e-08
      62149.71c -6.20315304662e-07
      54131.71c -2.24117871828e-05
      1001.71c -0.0171208583763
      62147.71c -4.09568278955e-06
      92234.71c -4.12636892683e-08
      92235.71c -0.0139951866635
      92236.71c -0.000271671386103
      43099.71c -3.65465234656e-05
      42095.71c -2.92118385994e-05
      61147.71c -1.49072209659e-05
      40090.71c -0.905572617977
      45103.71c -1.72897237071e-05
      55133.71c -5.34663621774e-05
      93237.71c -1.35887190154e-06
      55135.71c -3.69490425193e-05
      94240.71c -8.9873266319e-06
      94241.71c -9.003744523e-07
      94242.71c -5.42271201987e-08
m5234 62152.71c -4.80415788553e-06
      62151.71c -1.86098651647e-06
      62150.71c -9.8287751857e-06
      64155.71c -6.4400262318e-09
      63153.71c -1.78095694759e-06
      60143.71c -5.10475842219e-05
      60145.71c -3.64059898881e-05
      95241.71c -4.69541192274e-08
      95243.71c -1.24184943566e-09
      94239.71c -0.000140780213242
      94238.71c -4.22168196318e-08
      63154.71c -6.45960181678e-08
      44103.71c -1.84218909545e-06
      92238.71c -0.0625703479181
      54135.71c -3.25599143868e-08
      62149.71c -6.1796477793e-07
      54131.71c -2.36017817501e-05
      1001.71c -0.0171219100291
      62147.71c -4.30669683389e-06
      92234.71c -4.32681734415e-08
      92235.71c -0.01390609274
      92236.71c -0.00028605212503
      43099.71c -3.85068547169e-05
      42095.71c -3.07869804201e-05
      61147.71c -1.56804692135e-05
      40090.71c -0.90562824299
      45103.71c -1.82114685532e-05
      55133.71c -5.63306206969e-05
      93237.71c -1.49478642324e-06
      55135.71c -3.82506062365e-05
      94240.71c -9.88201982315e-06
      94241.71c -1.03040148489e-06
      94242.71c -6.54167106109e-08
c **  E-24  Fuel ID= 3111
m5241 62152.71c -3.44087296675e-06
      62151.71c -1.55056392138e-06
      62150.71c -7.17577868636e-06
      64155.71c -7.12517508898e-09
      63153.71c -1.29760521255e-06
      60143.71c -3.85714390616e-05
      60145.71c -2.72989347947e-05
      95241.71c -2.90974312973e-08
      95243.71c -2.81481290404e-10
      94239.71c -9.2998555052e-05
      94238.71c -2.03596684731e-08
      63154.71c -3.90021880589e-08
      44103.71c -1.17923135874e-06
      92238.71c -0.0626144328972
      54135.71c -2.39267417572e-08
      62149.71c -6.16888277165e-07
      54131.71c -1.77521101487e-05
      1001.71c -0.0171167133287
      62147.71c -4.07018578411e-06
      92234.71c -2.91904725085e-08
      92235.71c -0.0143485048603
      92236.71c -0.000211471451439
      43099.71c -2.88723585042e-05
      42095.71c -2.39680115489e-05
      61147.71c -1.10826830502e-05
      40090.71c -0.905353374203
      45103.71c -1.38125067327e-05
      55133.71c -4.22909103668e-05
      93237.71c -8.59627493892e-07
      55135.71c -3.23041308425e-05
      94240.71c -5.69043584748e-06
      94241.71c -4.96537186052e-07
      94242.71c -2.49090537161e-08
m5242 62152.71c -4.43451143796e-06
      62151.71c -1.72614928103e-06
      62150.71c -9.04518758372e-06
      64155.71c -7.1019891747e-09
      63153.71c -1.67786942165e-06
      60143.71c -4.7344200313e-05
      60145.71c -3.3704518882e-05
      95241.71c -6.56071505606e-08
      95243.71c -1.18025191902e-09
      94239.71c -0.000128465607008
      94238.71c -4.67175325205e-08
      63154.71c -6.31275550517e-08
      44103.71c -1.46788822963e-06
      92238.71c -0.0625776681847
      54135.71c -2.79502834846e-08
      62149.71c -6.19685622543e-07
      54131.71c -2.18924283307e-05
      1001.71c -0.0171203590505
      62147.71c -4.96977419876e-06
      92234.71c -4.03566690372e-08
      92235.71c -0.0140372305093
      92236.71c -0.000264501725177
      43099.71c -3.56907759073e-05
      42095.71c -2.96413628746e-05
      61147.71c -1.35911284819e-05
      40090.71c -0.905546207168
      45103.71c -1.7072933288e-05
      55133.71c -5.22622159971e-05
      93237.71c -1.48257817781e-06
      55135.71c -3.78868439691e-05
      94240.71c -9.60162408217e-06
      94241.71c -1.13191935169e-06
      94242.71c -7.21183897389e-08
m5243 62152.71c -5.42532948742e-06
      62151.71c -1.84593013092e-06
      62150.71c -1.08382893478e-05
      64155.71c -6.95418019629e-09
      63153.71c -2.04414086929e-06
      60143.71c -5.5578303504e-05
      60145.71c -3.9804156806e-05
      95241.71c -1.00056920283e-07
      95243.71c -2.6158941805e-09
      94239.71c -0.000148044151236
      94238.71c -7.58956798775e-08
      63154.71c -8.7774656674e-08
      44103.71c -1.73801011546e-06
      92238.71c -0.0625582926627
      54135.71c -3.11585862234e-08
      62149.71c -6.12066210184e-07
      54131.71c -2.58237655222e-05
      1001.71c -0.0171238515681
      62147.71c -5.82499169602e-06
      92234.71c -4.71809024751e-08
      92235.71c -0.0137426061733
      92236.71c -0.000311968443614
      43099.71c -4.21900354383e-05
      42095.71c -3.50664349568e-05
      61147.71c -1.5975603302e-05
      40090.71c -0.905730936708
      45103.71c -2.01513203184e-05
      55133.71c -6.17624449512e-05
      93237.71c -2.01706608873e-06
      55135.71c -4.24761294847e-05
      94240.71c -1.28958253623e-05
      94241.71c -1.74469815748e-06
      94242.71c -1.341146027e-07
m5244 62152.71c -5.7675509625e-06
      62151.71c -1.87975742886e-06
      62150.71c -1.14504736327e-05
      64155.71c -6.89337997711e-09
      63153.71c -2.17330803646e-06
      60143.71c -5.83473243385e-05
      60145.71c -4.18755575319e-05
      95241.71c -1.13061240055e-07
      95243.71c -3.3194021247e-09
      94239.71c -0.000154258136238
      94238.71c -8.79831894995e-08
      63154.71c -9.7017931956e-08
      44103.71c -1.83119490784e-06
      92238.71c -0.0625518257364
      54135.71c -3.21701461989e-08
      62149.71c -6.09336335819e-07
      54131.71c -2.71566672846e-05
      1001.71c -0.0171250441405
      62147.71c -6.11243424007e-06
      92234.71c -4.94085788417e-08
      92235.71c -0.0136426066448
      92236.71c -0.000328019842934
      43099.71c -4.43994308619e-05
      42095.71c -3.69102907692e-05
      61147.71c -1.67796519883e-05
      40090.71c -0.905794015373
      45103.71c -2.11941194687e-05
      55133.71c -6.49915094168e-05
      93237.71c -2.2182535566e-06
      55135.71c -4.39147190309e-05
      94240.71c -1.40876101814e-05
      94241.71c -1.97977645928e-06
      94242.71c -1.61306069189e-07
c **  F-1  Fuel ID= 5017
m6011 62152.71c -5.87968034981e-06
      62151.71c -1.9824175599e-06
      62150.71c -1.16892487644e-05
      64155.71c -1.04344270509e-08
      63153.71c -2.05907079958e-06
      60143.71c -6.11045174557e-05
      60145.71c -4.29073798081e-05
      95241.71c -2.86286608797e-08
      95243.71c -1.36691896503e-10
      94239.71c -0.000144270814363
      94238.71c -2.76736782284e-08
      63154.71c -6.98680136409e-08
      44103.71c -1.21724079037e-06
      92238.71c -0.0625789175003
      54135.71c -2.41218407736e-08
      62149.71c -5.93644781529e-07
      54131.71c -2.80535654361e-05
      1001.71c -0.0171252621897
      62147.71c -7.61469234505e-06
      92234.71c -4.60205856669e-08
      92235.71c -0.0135950651443
      92236.71c -0.000332237065813
      43099.71c -4.54886087072e-05
      42095.71c -4.08491663925e-05
      61147.71c -1.63067730496e-05
      40090.71c -0.905805548636
      45103.71c -2.23549997508e-05
      55133.71c -6.6772091506e-05
      93237.71c -1.49326084933e-06
      55135.71c -5.08231135814e-05
      94240.71c -1.0516842922e-05
      94241.71c -7.63099863669e-07
      94242.71c -2.23514513564e-08
m6012 62152.71c -7.64807771521e-06
      62151.71c -2.10352309388e-06
      62150.71c -1.47300395784e-05
      64155.71c -9.76803796057e-09
      63153.71c -2.69755747186e-06
      60143.71c -7.52900435613e-05
      60145.71c -5.32275875806e-05
      95241.71c -7.01272877188e-08
      95243.71c -6.29049460386e-10
      94239.71c -0.000197354529984
      94238.71c -6.45054764371e-08
      63154.71c -1.14622975724e-07
      44103.71c -1.51723520821e-06
      92238.71c -0.0625200423319
      54135.71c -2.79331706204e-08
      62149.71c -5.91199803361e-07
      54131.71c -3.47597439525e-05
      1001.71c -0.0171311421622
      62147.71c -9.39649537481e-06
      92234.71c -6.34211490974e-08
      92235.71c -0.0130947982808
      92236.71c -0.000417175095685
      43099.71c -5.65145137609e-05
      42095.71c -5.08289108342e-05
      61147.71c -2.00001020114e-05
      40090.71c -0.906116557699
      45103.71c -2.77842540253e-05
      55133.71c -8.29300137741e-05
      93237.71c -2.62924470652e-06
      55135.71c -5.97953526429e-05
      94240.71c -1.82065176359e-05
      94241.71c -1.85822288233e-06
      94242.71c -7.02572889821e-08
m6013 62152.71c -9.34556700763e-06
      62151.71c -2.14992686435e-06
      62150.71c -1.75702772223e-05
      64155.71c -9.15607450548e-09
      63153.71c -3.30732707073e-06
      60143.71c -8.82154617596e-05
      60145.71c -6.27988081105e-05
      95241.71c -1.13203595904e-07
      95243.71c -1.4779917311e-09
      94239.71c -0.00022486540239
      94238.71c -1.05553285571e-07
      63154.71c -1.60446837652e-07
      44103.71c -1.7834936847e-06
      92238.71c -0.0624886543015
      54135.71c -3.07058408911e-08
      62149.71c -5.7867933636e-07
      54131.71c -4.09610442087e-05
      1001.71c -0.0171366383477
      62147.71c -1.10618318483e-05
      92234.71c -7.363417388e-08
      92235.71c -0.0126335185407
      92236.71c -0.000491323547278
      43099.71c -6.67547642732e-05
      42095.71c -6.01496707661e-05
      61147.71c -2.33926711668e-05
      40090.71c -0.906407267137
      45103.71c -3.27803030936e-05
      55133.71c -9.79316124881e-05
      93237.71c -3.61155823077e-06
      55135.71c -6.6946098542e-05
      94240.71c -2.47762417377e-05
      94241.71c -2.98571399995e-06
      94242.71c -1.37494442582e-07
m6014 62152.71c -9.92765863508e-06
      62151.71c -2.1586477786e-06
      62150.71c -1.85417378994e-05
      64155.71c -8.96588822516e-09
      63153.71c -3.52386938461e-06
      60143.71c -9.2558206358e-05
      60145.71c -6.60518616121e-05
      95241.71c -1.30429155625e-07
      95243.71c -1.9068256271e-09
      94239.71c -0.000233435603073
      94238.71c -1.22659943364e-07
      63154.71c -1.77745534094e-07
      44103.71c -1.87403249663e-06
      92238.71c -0.0624781277185
      54135.71c -3.15483364856e-08
      62149.71c -5.74327495321e-07
      54131.71c -4.30635362181e-05
      1001.71c -0.0171385186308
      62147.71c -1.16272419115e-05
      92234.71c -7.70605052791e-08
      92235.71c -0.0124768287363
      92236.71c -0.000516384467541
      43099.71c -7.0240466635e-05
      42095.71c -6.33298578583e-05
      61147.71c -2.45301603658e-05
      40090.71c -0.906506720848
      45103.71c -3.44759790812e-05
      55133.71c -0.000103034505045
      93237.71c -3.97601656121e-06
      55135.71c -6.91841190816e-05
      94240.71c -2.71575480566e-05
      94241.71c -3.43572322463e-06
      94242.71c -1.68183643824e-07
c **  F-2  Fuel ID= 5018
m6021 62152.71c -5.63847442732e-06
      62151.71c -1.96506866131e-06
      62150.71c -1.12794296292e-05
      64155.71c -1.02645494313e-08
      63153.71c -1.98122972098e-06
      60143.71c -5.9117662452e-05
      60145.71c -4.14953749619e-05
      95241.71c -2.51887311207e-08
      95243.71c -1.15214638902e-10
      94239.71c -0.00014027700755
      94238.71c -2.51121051037e-08
      63154.71c -6.54872525694e-08
      44103.71c -1.21442640687e-06
      92238.71c -0.0625821552209
      54135.71c -2.41048163104e-08
      62149.71c -5.95843658259e-07
      54131.71c -2.71257862397e-05
      1001.71c -0.0171244812327
      62147.71c -7.18030237922e-06
      92234.71c -4.45547924816e-08
      92235.71c -0.0136631315877
      92236.71c -0.00032138896933
      43099.71c -4.39801159582e-05
      42095.71c -3.93117675956e-05
      61147.71c -1.59593486098e-05
      40090.71c -0.905764241522
      45103.71c -2.15847393423e-05
      55133.71c -6.45514835797e-05
      93237.71c -1.40303850916e-06
      55135.71c -4.91644028425e-05
      94240.71c -9.86801283137e-06
      94241.71c -6.93576222104e-07
      94242.71c -1.95483379831e-08
m6022 62152.71c -7.34547183049e-06
      62151.71c -2.09182757684e-06
      62150.71c -1.42245634051e-05
      64155.71c -9.63021046185e-09
      63153.71c -2.59210620541e-06
      60143.71c -7.28987365771e-05
      60145.71c -5.15030107969e-05
      95241.71c -6.18969153381e-08
      95243.71c -5.31851029521e-10
      94239.71c -0.000192253866362
      94238.71c -5.84781223162e-08
      63154.71c -1.07472680049e-07
      44103.71c -1.51429037959e-06
      92238.71c -0.0625252499662
      54135.71c -2.79672373567e-08
      62149.71c -5.93874436034e-07
      54131.71c -3.36299591298e-05
      1001.71c -0.0171301778062
      62147.71c -8.8668007224e-06
      92234.71c -6.14624115878e-08
      92235.71c -0.0131778818895
      92236.71c -0.000403822876338
      43099.71c -5.46665390198e-05
      42095.71c -4.89400411848e-05
      61147.71c -1.95912278078e-05
      40090.71c -0.906065550073
      45103.71c -2.68414678588e-05
      55133.71c -8.0211088548e-05
      93237.71c -2.46982489904e-06
      55135.71c -5.7874507291e-05
      94240.71c -1.71250403271e-05
      94241.71c -1.69409711246e-06
      94242.71c -6.16080182563e-08
m6023 62152.71c -8.98919354818e-06
      62151.71c -2.14346234042e-06
      62150.71c -1.69797050349e-05
      64155.71c -9.0337584086e-09
      63153.71c -3.17604441387e-06
      60143.71c -8.54856803854e-05
      60145.71c -6.07986948397e-05
      95241.71c -1.00178261411e-07
      95243.71c -1.25292405648e-09
      94239.71c -0.000219469796512
      94238.71c -9.56643405236e-08
      63154.71c -1.50543722893e-07
      44103.71c -1.78137889978e-06
      92238.71c -0.0624952105381
      54135.71c -3.07737954606e-08
      62149.71c -5.81937587909e-07
      54131.71c -3.96547204683e-05
      1001.71c -0.0171355030016
      62147.71c -1.04447320184e-05
      92234.71c -7.14060485609e-08
      92235.71c -0.0127298359653
      92236.71c -0.000475896991956
      43099.71c -6.46066433892e-05
      42095.71c -5.79436954847e-05
      61147.71c -2.29350709973e-05
      40090.71c -0.906347215336
      45103.71c -3.16879919387e-05
      55133.71c -9.47726722402e-05
      93237.71c -3.39329022913e-06
      55135.71c -6.48336270726e-05
      94240.71c -2.3349832032e-05
      94241.71c -2.73030866629e-06
      94242.71c -1.20835669033e-07
m6024 62152.71c -9.55354512267e-06
      62151.71c -2.15377176407e-06
      62150.71c -1.79224486312e-05
      64155.71c -8.84685232701e-09
      63153.71c -3.38346138166e-06
      60143.71c -8.97185470324e-05
      60145.71c -6.39594638089e-05
      95241.71c -1.15519510998e-07
      95243.71c -1.61760660319e-09
      94239.71c -0.000227962227814
      94238.71c -1.11163581223e-07
      63154.71c -1.6680800352e-07
      44103.71c -1.87224316029e-06
      92238.71c -0.0624850986845
      54135.71c -3.16311096521e-08
      62149.71c -5.77693408281e-07
      54131.71c -4.16984910048e-05
      1001.71c -0.0171373262813
      62147.71c -1.09810212398e-05
      92234.71c -7.47520579737e-08
      92235.71c -0.0125775756141
      92236.71c -0.00050027068386
      43099.71c -6.79914922408e-05
      42095.71c -6.10181070783e-05
      61147.71c -2.40571644694e-05
      40090.71c -0.90644365397
      45103.71c -3.33340212134e-05
      55133.71c -9.97281852868e-05
      93237.71c -3.73655240766e-06
      55135.71c -6.70138333087e-05
      94240.71c -2.56091720854e-05
      94241.71c -3.14508520771e-06
      94242.71c -1.478995201e-07
c **  F-3  Fuel ID= 5027
m6031 62152.71c -5.38230189226e-06
      62151.71c -1.94581167289e-06
      62150.71c -1.0844434777e-05
      64155.71c -1.01952465152e-08
      63153.71c -1.9008734337e-06
      60143.71c -5.70119050565e-05
      60145.71c -3.99956976072e-05
      95241.71c -2.242243773e-08
      95243.71c -9.76598733237e-11
      94239.71c -0.000136653135801
      94238.71c -2.28930841625e-08
      63154.71c -6.11565079289e-08
      44103.71c -1.20053843201e-06
      92238.71c -0.0625848693717
      54135.71c -2.39407936744e-08
      62149.71c -5.98698791606e-07
      54131.71c -2.61420194925e-05
      1001.71c -0.0171236475873
      62147.71c -6.78651036237e-06
      92234.71c -4.32210049461e-08
      92235.71c -0.0137353001904
      92236.71c -0.000310070347243
      43099.71c -4.23795623666e-05
      42095.71c -3.77556761783e-05
      61147.71c -1.55207184777e-05
      40090.71c -0.905720147562
      45103.71c -2.07768455724e-05
      55133.71c -6.2197540819e-05
      93237.71c -1.3252267966e-06
      55135.71c -4.7478675598e-05
      94240.71c -9.2360704467e-06
      94241.71c -6.31693678324e-07
      94242.71c -1.70772410316e-08
m6032 62152.71c -7.07066919489e-06
      62151.71c -2.08132708527e-06
      62150.71c -1.37688358054e-05
      64155.71c -9.56646333566e-09
      63153.71c -2.5008607284e-06
      60143.71c -7.07396723408e-05
      60145.71c -4.99455688035e-05
      95241.71c -5.61757577192e-08
      95243.71c -4.64679244275e-10
      94239.71c -0.000188662259692
      94238.71c -5.39091127406e-08
      63154.71c -1.01559064167e-07
      44103.71c -1.50596145855e-06
      92238.71c -0.0625288373759
      54135.71c -2.78954112579e-08
      62149.71c -5.96929612037e-07
      54131.71c -3.26096914277e-05
      1001.71c -0.0171293032978
      62147.71c -8.43536368229e-06
      92234.71c -5.97373726162e-08
      92235.71c -0.013252839908
      92236.71c -0.000391906971229
      43099.71c -5.2999637614e-05
      42095.71c -4.72924753555e-05
      61147.71c -1.91706720373e-05
      40090.71c -0.906019294748
      45103.71c -2.59960563249e-05
      55133.71c -7.77603810347e-05
      93237.71c -2.35362218729e-06
      55135.71c -5.61544547325e-05
      94240.71c -1.62354342517e-05
      94241.71c -1.57335986865e-06
      94242.71c -5.51579939372e-08
m6033 62152.71c -8.68778440023e-06
      62151.71c -2.13849548402e-06
      62150.71c -1.64866686949e-05
      64155.71c -8.96866661233e-09
      63153.71c -3.07086131855e-06
      60143.71c -8.31978768058e-05
      60145.71c -5.91242160477e-05
      95241.71c -9.10403468626e-08
      95243.71c -1.10395298693e-09
      94239.71c -0.000215462246755
      94238.71c -8.87725351525e-08
      63154.71c -1.42970506034e-07
      44103.71c -1.77460861348e-06
      92238.71c -0.0624998517361
      54135.71c -3.07614232559e-08
      62149.71c -5.85166989945e-07
      54131.71c -3.85595572559e-05
      1001.71c -0.01713455795
      62147.71c -9.96454782589e-06
      92234.71c -6.96845410201e-08
      92235.71c -0.0128102281732
      92236.71c -0.000463144305505
      43099.71c -6.28085084811e-05
      42095.71c -5.61556751991e-05
      61147.71c -2.25072548507e-05
      40090.71c -0.906297228774
      45103.71c -3.07763786377e-05
      55133.71c -9.21297889363e-05
      93237.71c -3.24991185624e-06
      55135.71c -6.3030675594e-05
      94240.71c -2.21962806112e-05
      94241.71c -2.54072633024e-06
      94242.71c -1.08528258605e-07
m6034 62152.71c -9.23159215668e-06
      62151.71c -2.15020806226e-06
      62150.71c -1.73975973102e-05
      64155.71c -8.7825448633e-09
      63153.71c -3.26957726193e-06
      60143.71c -8.73024143278e-05
      60145.71c -6.21808483041e-05
      95241.71c -1.05369590374e-07
      95243.71c -1.42823623954e-09
      94239.71c -0.000223665751093
      94238.71c -1.02998027948e-07
      63154.71c -1.58420607626e-07
      44103.71c -1.86505313037e-06
      92238.71c -0.0624902461497
      54135.71c -3.16227197105e-08
      62149.71c -5.81146571231e-07
      54131.71c -4.05357996422e-05
      1001.71c -0.0171363201275
      62147.71c -1.04754520291e-05
      92234.71c -7.32370431531e-08
      92235.71c -0.0126629052145
      92236.71c -0.000486773823774
      43099.71c -6.60802908093e-05
      42095.71c -5.91164526388e-05
      61147.71c -2.36018076126e-05
      40090.71c -0.906390435534
      45103.71c -3.23636490721e-05
      55133.71c -9.69203666129e-05
      93237.71c -3.57116657185e-06
      55135.71c -6.51417058077e-05
      94240.71c -2.43190334898e-05
      94241.71c -2.93414213652e-06
      94242.71c -1.33237241845e-07
c **  F-4  Fuel ID= 5021
m6041 62152.71c -5.61177073085e-06
      62151.71c -1.96300944705e-06
      62150.71c -1.12338519797e-05
      64155.71c -1.02446640896e-08
      63153.71c -1.97263772941e-06
      60143.71c -5.88963113081e-05
      60145.71c -4.13383422148e-05
      95241.71c -2.48318193627e-08
      95243.71c -1.13065691706e-10
      94239.71c -0.00013982821544
      94238.71c -2.48412154879e-08
      63154.71c -6.50114808754e-08
      44103.71c -1.21415242766e-06
      92238.71c -0.0625825319441
      54135.71c -2.41075583264e-08
      62149.71c -5.96094797852e-07
      54131.71c -2.70225783145e-05
      1001.71c -0.0171243940672
      62147.71c -7.13271916776e-06
      92234.71c -4.43911790741e-08
      92235.71c -0.0136707048495
      92236.71c -0.000320182327798
      43099.71c -4.38123657008e-05
      42095.71c -3.91407849114e-05
      61147.71c -1.59198925459e-05
      40090.71c -0.90575963108
      45103.71c -2.14990248323e-05
      55133.71c -6.43045157447e-05
      93237.71c -1.39321381706e-06
      55135.71c -4.89798564987e-05
      94240.71c -9.79741053523e-06
      94241.71c -6.86182965649e-07
      94242.71c -1.92593442385e-08
m6042 62152.71c -7.31185672533e-06
      62151.71c -2.0904069841e-06
      62150.71c -1.41682967802e-05
      64155.71c -9.61456041475e-09
      63153.71c -2.58047912725e-06
      60143.71c -7.26320884439e-05
      60145.71c -5.13109055486e-05
      95241.71c -6.10405517271e-08
      95243.71c -5.22076045887e-10
      94239.71c -0.000191680190494
      94238.71c -5.78409332655e-08
      63154.71c -1.06695114581e-07
      44103.71c -1.51387880674e-06
      92238.71c -0.0625258610837
      54135.71c -2.79687772399e-08
      62149.71c -5.94166136842e-07
      54131.71c -3.35041382639e-05
      1001.71c -0.0171300697866
      62147.71c -8.808726448e-06
      92234.71c -6.12435262883e-08
      92235.71c -0.0131871402435
      92236.71c -0.000402334787687
      43099.71c -5.44607853359e-05
      42095.71c -4.87298059467e-05
      61147.71c -1.95447254735e-05
      40090.71c -0.906059836594
      45103.71c -2.67365005341e-05
      55133.71c -7.99084257876e-05
      93237.71c -2.45250027114e-06
      55135.71c -5.7660463057e-05
      94240.71c -1.70069227982e-05
      94241.71c -1.67660323592e-06
      94242.71c -6.07133083193e-08
m6043 62152.71c -8.94955516684e-06
      62151.71c -2.14263821451e-06
      62150.71c -1.69139486428e-05
      64155.71c -9.01968916154e-09
      63153.71c -3.1615747323e-06
      60143.71c -8.5180778373e-05
      60145.71c -6.05758151719e-05
      95241.71c -9.88191411162e-08
      95243.71c -1.23021051002e-09
      94239.71c -0.000218859196997
      94238.71c -9.46186631189e-08
      63154.71c -1.49465939321e-07
      44103.71c -1.78114554938e-06
      92238.71c -0.0624959283021
      54135.71c -3.07846456184e-08
      62149.71c -5.82276510811e-07
      54131.71c -3.95090777131e-05
      1001.71c -0.0171353768685
      62147.71c -1.03770184333e-05
      92234.71c -7.11578217172e-08
      92235.71c -0.0127405675797
      92236.71c -0.000474176818754
      43099.71c -6.43673331518e-05
      42095.71c -5.76979261689e-05
      61147.71c -2.2882883216e-05
      40090.71c -0.906340543788
      45103.71c -3.1566209615e-05
      55133.71c -9.44207391523e-05
      93237.71c -3.36955994541e-06
      55135.71c -6.45980766068e-05
      94240.71c -2.31937069096e-05
      94241.71c -2.70297719539e-06
      94242.71c -1.19109414383e-07
m6044 62152.71c -9.51190187388e-06
      62151.71c -2.15312720034e-06
      62150.71c -1.78534476022e-05
      64155.71c -8.83333653409e-09
      63153.71c -3.36797670492e-06
      60143.71c -8.94011687854e-05
      60145.71c -6.37262213248e-05
      95241.71c -1.1396245679e-07
      95243.71c -1.5883711449e-09
      94239.71c -0.000227341153447
      94238.71c -1.0994780694e-07
      63154.71c -1.65616977328e-07
      44103.71c -1.87200929346e-06
      92238.71c -0.0624858495253
      54135.71c -3.16460084058e-08
      62149.71c -5.78039703477e-07
      54131.71c -4.15461889908e-05
      1001.71c -0.0171371940263
      62147.71c -1.09100957651e-05
      92234.71c -7.44958114818e-08
      92235.71c -0.0125888052013
      92236.71c -0.000498472875674
      43099.71c -6.7740829113e-05
      42095.71c -6.07605615188e-05
      61147.71c -2.40031148317e-05
      40090.71c -0.906436658613
      45103.71c -3.32066694997e-05
      55133.71c -9.93595871435e-05
      93237.71c -3.7104843428e-06
      55135.71c -6.67717209672e-05
      94240.71c -2.54397211624e-05
      94241.71c -3.11385197963e-06
      94242.71c -1.45796358836e-07
c **  F-5  Fuel ID= 5026
m6051 62152.71c -5.5842510647e-06
      62151.71c -1.96086234373e-06
      62150.71c -1.11868449035e-05
      64155.71c -1.02239759618e-08
      63153.71c -1.96378675762e-06
      60143.71c -5.8667944079e-05
      60145.71c -4.1176385127e-05
      95241.71c -2.44680608241e-08
      95243.71c -1.10889064643e-10
      94239.71c -0.000139364669417
      94238.71c -2.45641823754e-08
      63154.71c -6.45227183619e-08
      44103.71c -1.21387795054e-06
      92238.71c -0.0625829234749
      54135.71c -2.41118333245e-08
      62149.71c -5.96352830119e-07
      54131.71c -2.69161303033e-05
      1001.71c -0.0171243041154
      62147.71c -7.08376757217e-06
      92234.71c -4.42221633401e-08
      92235.71c -0.013678515961
      92236.71c -0.000318937785302
      43099.71c -4.36393515241e-05
      42095.71c -3.89644340281e-05
      61147.71c -1.58790551504e-05
      40090.71c -0.905754873267
      45103.71c -2.14106125091e-05
      55133.71c -6.40497903488e-05
      93237.71c -1.3831150253e-06
      55135.71c -4.87895020948e-05
      94240.71c -9.72485753382e-06
      94241.71c -6.78617091847e-07
      94242.71c -1.89651340696e-08
m6052 62152.71c -7.27719017486e-06
      62151.71c -2.08892034203e-06
      62150.71c -1.41102545617e-05
      64155.71c -9.59839052454e-09
      63153.71c -2.56850460719e-06
      60143.71c -7.2356969622e-05
      60145.71c -5.11126993817e-05
      95241.71c -6.01673287902e-08
      95243.71c -5.12165964979e-10
      94239.71c -0.000191087594509
      94238.71c -5.71893265909e-08
      63154.71c -1.05896046649e-07
      44103.71c -1.51343024345e-06
      92238.71c -0.0625264947614
      54135.71c -2.79694224061e-08
      62149.71c -5.94465147849e-07
      54131.71c -3.33743359204e-05
      1001.71c -0.0171299582673
      62147.71c -8.74897389695e-06
      92234.71c -6.1017449288e-08
      92235.71c -0.0131966931541
      92236.71c -0.000400799359947
      43099.71c -5.42485221506e-05
      42095.71c -4.8512936814e-05
      61147.71c -1.9496591228e-05
      40090.71c -0.906053938014
      45103.71c -2.66282239833e-05
      55133.71c -7.95962202821e-05
      93237.71c -2.43470116482e-06
      55135.71c -5.74396130329e-05
      94240.71c -1.68854469523e-05
      94241.71c -1.65869710218e-06
      94242.71c -5.98017939861e-08
m6053 62152.71c -8.90867547142e-06
      62151.71c -2.14176981275e-06
      62150.71c -1.68461204043e-05
      64155.71c -9.00506209117e-09
      63153.71c -3.1466734764e-06
      60143.71c -8.48660837261e-05
      60145.71c -6.03458797064e-05
      95241.71c -9.74325634306e-08
      95243.71c -1.20717597025e-09
      94239.71c -0.0002182275038
      94238.71c -9.35493989091e-08
      63154.71c -1.48358418277e-07
      44103.71c -1.78091530291e-06
      92238.71c -0.0624966639968
      54135.71c -3.07974029555e-08
      62149.71c -5.82617340329e-07
      54131.71c -3.93588017741e-05
      1001.71c -0.0171352468651
      62147.71c -1.03073384238e-05
      92234.71c -7.09016698527e-08
      92235.71c -0.0127516385171
      92236.71c -0.000472401962351
      43099.71c -6.41204440682e-05
      42095.71c -5.74443628201e-05
      61147.71c -2.2828835263e-05
      40090.71c -0.906333667521
      45103.71c -3.14405543656e-05
      55133.71c -9.40576562763e-05
      93237.71c -3.34518348492e-06
      55135.71c -6.43550295875e-05
      94240.71c -2.30331222625e-05
      94241.71c -2.67496764415e-06
      94242.71c -1.17350514413e-07
m6054 62152.71c -9.4689409652e-06
      62151.71c -2.15244471372e-06
      62150.71c -1.77822611953e-05
      64155.71c -8.81935488121e-09
      63153.71c -3.35202985136e-06
      60143.71c -8.90735605865e-05
      60145.71c -6.34855615093e-05
      95241.71c -1.12373745869e-07
      95243.71c -1.55872712425e-09
      94239.71c -0.00022669856489
      94238.71c -1.08704581243e-07
      63154.71c -1.64392807945e-07
      44103.71c -1.8717589366e-06
      92238.71c -0.0624866159817
      54135.71c -3.16628867216e-08
      62149.71c -5.78388133706e-07
      54131.71c -4.13890049127e-05
      1001.71c -0.0171370577447
      62147.71c -1.0837108508e-05
      92234.71c -7.42319170821e-08
      92235.71c -0.0126003918294
      92236.71c -0.000496617635296
      43099.71c -6.7482196193e-05
      42095.71c -6.04948581726e-05
      61147.71c -2.39471214189e-05
      40090.71c -0.906429450279
      45103.71c -3.30752628382e-05
      55133.71c -9.89792514621e-05
      93237.71c -3.6836932358e-06
      55135.71c -6.6521879168e-05
      94240.71c -2.52654239144e-05
      94241.71c -3.08182260583e-06
      94242.71c -1.43653033653e-07
c **  F-6  Fuel ID= 6223
m6061 62152.71c -9.47270755314e-07
      62151.71c -7.61025375647e-07
      62150.71c -2.21484736825e-06
      64155.71c -2.69889158848e-09
      63153.71c -4.0769920758e-07
      60143.71c -1.30760685146e-05
      60145.71c -9.35359100122e-06
      95241.71c -2.3091515246e-10
      95243.71c -2.86444611612e-13
      94239.71c -3.49318667752e-05
      94238.71c -8.2804016017e-10
      63154.71c -5.15137519882e-09
      44103.71c -7.74682851454e-07
      92238.71c -0.0626535187201
      54135.71c -1.60994930209e-08
      62149.71c -4.34519616603e-07
      54131.71c -6.00890565852e-06
      1001.71c -0.0171068245346
      62147.71c -6.19454032071e-07
      92234.71c -1.03716985879e-08
      92235.71c -0.0152153514018
      92236.71c -7.28953649057e-05
      43099.71c -9.81813063215e-06
      42095.71c -6.40825945986e-06
      61147.71c -4.53064963806e-06
      40090.71c -0.904830326763
      45103.71c -4.37344051666e-06
      55133.71c -1.43033654861e-05
      93237.71c -1.36808173653e-07
      55135.71c -1.1150976209e-05
      94240.71c -7.77517944762e-07
      94241.71c -1.85984252652e-08
      94242.71c -1.56933072841e-10
m6062 62152.71c -1.26564831738e-06
      62151.71c -8.96071845503e-07
      62150.71c -2.91526346853e-06
      64155.71c -2.88350873662e-09
      63153.71c -5.22844485517e-07
      60143.71c -1.64894360092e-05
      60145.71c -1.18176849686e-05
      95241.71c -6.14190238974e-10
      95243.71c -1.44613543404e-12
      94239.71c -5.01810040247e-05
      94238.71c -1.84600628258e-09
      63154.71c -8.67026765448e-09
      44103.71c -9.81345932195e-07
      92238.71c -0.0626408593772
      54135.71c -1.91788347533e-08
      62149.71c -4.39419826815e-07
      54131.71c -7.5913635963e-06
      1001.71c -0.0171082091353
      62147.71c -7.81229295964e-07
      92234.71c -1.44910944217e-08
      92235.71c -0.0150950750835
      92236.71c -9.34705170386e-05
      43099.71c -1.24117404264e-05
      42095.71c -8.11021608884e-06
      61147.71c -5.70343339276e-06
      40090.71c -0.904903562374
      45103.71c -5.53388295113e-06
      55133.71c -1.80800506766e-05
      93237.71c -2.33463901514e-07
      55135.71c -1.33345357069e-05
      94240.71c -1.43723472873e-06
      94241.71c -4.94228984413e-08
      94242.71c -5.35033655012e-10
m6063 62152.71c -1.58583730708e-06
      62151.71c -1.00049698003e-06
      62150.71c -3.57931971481e-06
      64155.71c -2.95828938651e-09
      63153.71c -6.31647352093e-07
      60143.71c -1.96757682704e-05
      60145.71c -1.4127496494e-05
      95241.71c -1.04528485065e-09
      95243.71c -3.58465008121e-12
      94239.71c -5.93930745534e-05
      94238.71c -2.95717000957e-09
      63154.71c -1.23848931112e-08
      44103.71c -1.1732776086e-06
      92238.71c -0.0626343928318
      54135.71c -2.16559993873e-08
      62149.71c -4.37160405475e-07
      54131.71c -9.07294457488e-06
      1001.71c -0.0171095100972
      62147.71c -9.33155263105e-07
      92234.71c -1.72036509423e-08
      92235.71c -0.0149832052007
      92236.71c -0.000111688233211
      43099.71c -1.48438977379e-05
      42095.71c -9.71178898769e-06
      61147.71c -6.80004436056e-06
      40090.71c -0.904972374079
      45103.71c -6.61946220773e-06
      55133.71c -2.16213520057e-05
      93237.71c -3.16959577126e-07
      55135.71c -1.51219296492e-05
      94240.71c -2.04058152523e-06
      94241.71c -8.40572129138e-08
      94242.71c -1.09769301308e-09
m6064 62152.71c -1.7009863793e-06
      62151.71c -1.03305151283e-06
      62150.71c -3.81083849745e-06
      64155.71c -2.97216852243e-09
      63153.71c -6.70075625262e-07
      60143.71c -2.07807123193e-05
      60145.71c -1.49305132232e-05
      95241.71c -1.22951837146e-09
      95243.71c -4.7198507359e-12
      94239.71c -6.24244410308e-05
      94238.71c -3.42545927063e-09
      63154.71c -1.38220735249e-08
      44103.71c -1.23995563667e-06
      92238.71c -0.0626321181216
      54135.71c -2.24475830206e-08
      62149.71c -4.36184934062e-07
      54131.71c -9.58778483632e-06
      1001.71c -0.0171099664864
      62147.71c -9.85949488874e-07
      92234.71c -1.82104158217e-08
      92235.71c -0.0149443206144
      92236.71c -0.00011801039442
      43099.71c -1.56899068916e-05
      42095.71c -1.02699598e-05
      61147.71c -7.18029475053e-06
      40090.71c -0.904996513847
      45103.71c -6.99712554349e-06
      55133.71c -2.2853133142e-05
      93237.71c -3.48511906241e-07
      55135.71c -1.56986490341e-05
      94240.71c -2.27013183752e-06
      94241.71c -9.88487908535e-08
      94242.71c -1.36877430431e-09
c **  F-7  Fuel ID= 5256
m6071 62152.71c -6.34502183103e-06
      62151.71c -2.01143967114e-06
      62150.71c -1.24738186711e-05
      64155.71c -1.0729316275e-08
      63153.71c -2.21034171022e-06
      60143.71c -6.48927201548e-05
      60145.71c -4.56089404193e-05
      95241.71c -3.6189020504e-08
      95243.71c -1.8714935719e-10
      94239.71c -0.000151751952132
      94238.71c -3.31021045165e-08
      63154.71c -7.86486024996e-08
      44103.71c -1.22299918129e-06
      92238.71c -0.0625724335208
      54135.71c -2.4113040635e-08
      62149.71c -5.89279916604e-07
      54131.71c -2.98274211244e-05
      1001.71c -0.0171267648344
      62147.71c -8.47038806341e-06
      92234.71c -4.87987640644e-08
      92235.71c -0.0134647915429
      92236.71c -0.000352970571753
      43099.71c -4.83761033222e-05
      42095.71c -4.37898940702e-05
      61147.71c -1.69443870418e-05
      40090.71c -0.905885027941
      45103.71c -2.38271357443e-05
      55133.71c -7.10228078757e-05
      93237.71c -1.67433562151e-06
      55135.71c -5.3992274113e-05
      94240.71c -1.18113897996e-05
      94241.71c -9.08577924708e-07
      94242.71c -2.8592473203e-08
m6072 62152.71c -8.228280536e-06
      62151.71c -2.12200684008e-06
      62150.71c -1.56951176344e-05
      64155.71c -1.00158218199e-08
      63153.71c -2.90286538449e-06
      60143.71c -7.98330437925e-05
      60145.71c -5.65174613017e-05
      95241.71c -8.81030378502e-08
      95243.71c -8.56376532698e-10
      94239.71c -0.000206812638838
      94238.71c -7.72707929429e-08
      63154.71c -1.28931097554e-07
      44103.71c -1.52162223909e-06
      92238.71c -0.062509645905
      54135.71c -2.78305562814e-08
      62149.71c -5.86074348494e-07
      54131.71c -3.69135831995e-05
      1001.71c -0.0171329937248
      62147.71c -1.04380690217e-05
      92234.71c -6.71523615002e-08
      92235.71c -0.0129363145394
      92236.71c -0.00044261601196
      43099.71c -6.00420163785e-05
      42095.71c -5.44371812453e-05
      61147.71c -2.07445617672e-05
      40090.71c -0.906214492299
      45103.71c -2.95824548314e-05
      55133.71c -8.81198270474e-05
      93237.71c -2.94804187047e-06
      55135.71c -6.34559278479e-05
      94240.71c -2.03488816301e-05
      94241.71c -2.1983130885e-06
      94242.71c -8.93912354405e-08
m6073 62152.71c -1.00266266906e-05
      62151.71c -2.15898339673e-06
      62150.71c -1.8696613818e-05
      64155.71c -9.37893878509e-09
      63153.71c -3.5630912884e-06
      60143.71c -9.338941527e-05
      60145.71c -6.66101724169e-05
      95241.71c -1.41495726009e-07
      95243.71c -2.00337007539e-09
      94239.71c -0.000234837833179
      94238.71c -1.26504203103e-07
      63154.71c -1.8024829955e-07
      44103.71c -1.78683618683e-06
      92238.71c -0.0624762206055
      54135.71c -3.05420609006e-08
      62149.71c -5.72756552077e-07
      54131.71c -4.34478927185e-05
      1001.71c -0.0171388037079
      62147.71c -1.2273298158e-05
      92234.71c -7.78733010742e-08
      92235.71c -0.0124500275279
      92236.71c -0.000520679424703
      43099.71c -7.08516180973e-05
      42095.71c -6.43563891574e-05
      61147.71c -2.4222909641e-05
      40090.71c -0.906521799412
      45103.71c -3.48602928089e-05
      55133.71c -0.00010395513202
      93237.71c -4.04728641137e-06
      55135.71c -7.09663710189e-05
      94240.71c -2.75900704884e-05
      94241.71c -3.51347299309e-06
      94242.71c -1.74214031568e-07
m6074 62152.71c -1.06414809576e-05
      62151.71c -2.16495733049e-06
      62150.71c -1.97224473762e-05
      64155.71c -9.18674281982e-09
      63153.71c -3.79752940256e-06
      60143.71c -9.79337500288e-05
      60145.71c -7.00358986803e-05
      95241.71c -1.62752439012e-07
      95243.71c -2.58204513695e-09
      94239.71c -0.000243478436525
      94238.71c -1.47010656995e-07
      63154.71c -1.99599579063e-07
      44103.71c -1.87663419871e-06
      92238.71c -0.0624651079419
      54135.71c -3.13581271809e-08
      62149.71c -5.68004549113e-07
      54131.71c -4.56598633156e-05
      1001.71c -0.0171407874305
      62147.71c -1.28951947683e-05
      92234.71c -8.14438778708e-08
      92235.71c -0.0122850887994
      92236.71c -0.000547028168799
      43099.71c -7.45263493116e-05
      42095.71c -6.77367619694e-05
      61147.71c -2.53867513627e-05
      40090.71c -0.906626724342
      45103.71c -3.66488817649e-05
      55133.71c -0.000109334719024
      93237.71c -4.45473003688e-06
      55135.71c -7.33115900864e-05
      94240.71c -3.02065034791e-05
      94241.71c -4.03607222786e-06
      94242.71c -2.12827830519e-07
c **  F-8  Fuel ID= 6525
m6081 62152.71c -1.08019380854e-06
      62151.71c -9.28954338059e-07
      62150.71c -2.48324429511e-06
      64155.71c -2.9310339285e-09
      63153.71c -4.77268130017e-07
      60143.71c -1.51637557879e-05
      60145.71c -1.10096555378e-05
      95241.71c -2.16806522346e-10
      95243.71c -2.72748757446e-13
      94239.71c -4.11715052788e-05
      94238.71c -8.24526155417e-10
      63154.71c -5.35718740094e-09
      44103.71c -1.13443307634e-06
      92238.71c -0.0626503840366
      54135.71c -2.39166484179e-08
      62149.71c -6.26946780618e-07
      54131.71c -7.01461742612e-06
      1001.71c -0.01710776136
      62147.71c -6.33027705862e-07
      92234.71c -1.21322548734e-08
      92235.71c -0.0151350407551
      92236.71c -8.5800941577e-05
      43099.71c -1.15226036663e-05
      42095.71c -6.74470298743e-06
      61147.71c -5.36613003669e-06
      40090.71c -0.904879878216
      45103.71c -4.92611922086e-06
      55133.71c -1.67311198634e-05
      93237.71c -1.47633533212e-07
      55135.71c -1.31020431e-05
      94240.71c -8.07087363305e-07
      94241.71c -1.81202031858e-08
      94242.71c -1.49943065745e-10
m6082 62152.71c -1.42669998311e-06
      62151.71c -1.09903133562e-06
      62150.71c -3.27037451297e-06
      64155.71c -3.17015082801e-09
      63153.71c -6.06788320707e-07
      60143.71c -1.90048598845e-05
      60145.71c -1.38205508184e-05
      95241.71c -5.6672470622e-10
      95243.71c -1.33351238277e-12
      94239.71c -5.88527127295e-05
      94238.71c -1.81011355087e-09
      63154.71c -8.90000694841e-09
      44103.71c -1.42736620781e-06
      92238.71c -0.062635926087
      54135.71c -2.83757475974e-08
      62149.71c -6.41612314996e-07
      54131.71c -8.80535917992e-06
      1001.71c -0.0171093447992
      62147.71c -7.93312725088e-07
      92234.71c -1.71345823897e-08
      92235.71c -0.0149976977126
      92236.71c -0.000109357592549
      43099.71c -1.44717008198e-05
      42095.71c -8.47980041349e-06
      61147.71c -6.71538125532e-06
      40090.71c -0.904963630978
      45103.71c -6.19246205772e-06
      55133.71c -2.10117490422e-05
      93237.71c -2.48143503185e-07
      55135.71c -1.5593926863e-05
      94240.71c -1.47322632342e-06
      94241.71c -4.73157138268e-08
      94242.71c -4.98266230489e-10
m6083 62152.71c -1.77975221213e-06
      62151.71c -1.2376840852e-06
      62150.71c -4.03399137297e-06
      64155.71c -3.29546996031e-09
      63153.71c -7.30896920736e-07
      60143.71c -2.2651075135e-05
      60145.71c -1.64985584224e-05
      95241.71c -9.62508590468e-10
      95243.71c -3.28897089989e-12
      94239.71c -6.98223571451e-05
      94238.71c -2.87032663002e-09
      63154.71c -1.26650287222e-08
      44103.71c -1.70499167285e-06
      92238.71c -0.0626282390948
      54135.71c -3.2050541873e-08
      62149.71c -6.42342564746e-07
      54131.71c -1.05095745347e-05
      1001.71c -0.0171108596625
      62147.71c -9.4605615536e-07
      92234.71c -2.01768271055e-08
      92235.71c -0.0148679143117
      92236.71c -0.000130518147703
      43099.71c -1.72824110085e-05
      42095.71c -1.0136793199e-05
      61147.71c -7.99849293734e-06
      40090.71c -0.905043756563
      45103.71c -7.39618760404e-06
      55133.71c -2.50907540964e-05
      93237.71c -3.33604903691e-07
      55135.71c -1.76711761907e-05
      94240.71c -2.09217289129e-06
      94241.71c -8.03050497254e-08
      94242.71c -1.01791550142e-09
m6084 62152.71c -1.90706664385e-06
      62151.71c -1.28175067867e-06
      62150.71c -4.30107589746e-06
      64155.71c -3.32476369282e-09
      63153.71c -7.74640043594e-07
      60143.71c -2.39186337356e-05
      60145.71c -1.7431473649e-05
      95241.71c -1.13045176979e-09
      95243.71c -4.30183807996e-12
      94239.71c -7.34292123667e-05
      94238.71c -3.30901276221e-09
      63154.71c -1.41201924219e-08
      44103.71c -1.8014491062e-06
      92238.71c -0.0626257353804
      54135.71c -3.32282353109e-08
      62149.71c -6.42004368243e-07
      54131.71c -1.11032905222e-05
      1001.71c -0.0171113886504
      62147.71c -9.9931220749e-07
      92234.71c -2.12757825668e-08
      92235.71c -0.0148227414801
      92236.71c -0.000137851319514
      43099.71c -1.82620283715e-05
      42095.71c -1.07159965734e-05
      61147.71c -8.44411313836e-06
      40090.71c -0.905071736291
      45103.71c -7.81600119317e-06
      55133.71c -2.65122821906e-05
      93237.71c -3.65265164303e-07
      55135.71c -1.83434888235e-05
      94240.71c -2.32590796396e-06
      94241.71c -9.42276298808e-08
      94242.71c -1.26576629147e-09
c **  F-9  Fuel ID= 5001
m6091 62152.71c -4.98414801489e-06
      62151.71c -1.90846321678e-06
      62150.71c -1.01540203612e-05
      64155.71c -9.74073832537e-09
      63153.71c -1.77163527176e-06
      60143.71c -5.36330770526e-05
      60145.71c -3.76159326415e-05
      95241.71c -1.73587897394e-08
      95243.71c -7.04747917488e-11
      94239.71c -0.000129028918434
      94238.71c -1.8968909913e-08
      63154.71c -5.41951333697e-08
      44103.71c -1.20702610493e-06
      92238.71c -0.0625912719376
      54135.71c -2.41225155792e-08
      62149.71c -6.01933329765e-07
      54131.71c -2.45745857389e-05
      1001.71c -0.0171223331081
      62147.71c -6.03724762013e-06
      92234.71c -4.0538486439e-08
      92235.71c -0.0138502366635
      92236.71c -0.000291559803377
      43099.71c -3.98379375609e-05
      42095.71c -3.50930266843e-05
      61147.71c -1.49485992107e-05
      40090.71c -0.905650620888
      45103.71c -1.94670675525e-05
      55133.71c -5.84521543633e-05
      93237.71c -1.1701397748e-06
      55135.71c -4.46021333689e-05
      94240.71c -8.18684595853e-06
      94241.71c -5.24453425378e-07
      94242.71c -1.3259148797e-08
m6092 62152.71c -6.51968473495e-06
      62151.71c -2.05117733551e-06
      62150.71c -1.28350458859e-05
      64155.71c -9.20829501922e-09
      63153.71c -2.30942343383e-06
      60143.71c -6.62811561178e-05
      60145.71c -4.67570346629e-05
      95241.71c -4.30184325247e-08
      95243.71c -3.28072833028e-10
      94239.71c -0.000177803489307
      94238.71c -4.40490138118e-08
      63154.71c -8.90321863688e-08
      44103.71c -1.50652855229e-06
      92238.71c -0.0625398991998
      54135.71c -2.80551809999e-08
      62149.71c -6.01269365617e-07
      54131.71c -3.05177454122e-05
      1001.71c -0.0171275210425
      62147.71c -7.46971230334e-06
      92234.71c -5.60451226341e-08
      92235.71c -0.0134066236369
      92236.71c -0.000367022502236
      43099.71c -4.95852352633e-05
      42095.71c -4.3746236874e-05
      61147.71c -1.83972381373e-05
      40090.71c -0.905925026015
      45103.71c -2.42446540913e-05
      55133.71c -7.27340676808e-05
      93237.71c -2.0596620107e-06
      55135.71c -5.25811293006e-05
      94240.71c -1.43025586046e-05
      94241.71c -1.29273119449e-06
      94242.71c -4.20869051491e-08
m6093 62152.71c -8.01139154437e-06
      62151.71c -2.11821301517e-06
      62150.71c -1.53533406929e-05
      64155.71c -8.66087595542e-09
      63153.71c -2.82450008462e-06
      60143.71c -7.79009088303e-05
      60145.71c -5.52821845303e-05
      95241.71c -7.00966607276e-08
      95243.71c -7.77165003646e-10
      94239.71c -0.000203959950123
      94238.71c -7.19812657573e-08
      63154.71c -1.24943381141e-07
      44103.71c -1.77748289367e-06
      92238.71c -0.062512763847
      54135.71c -3.10006209433e-08
      62149.71c -5.90408328881e-07
      54131.71c -3.60460369188e-05
      1001.71c -0.0171323886182
      62147.71c -8.81368015427e-06
      92234.71c -6.52391890147e-08
      92235.71c -0.012995550126
      92236.71c -0.000433270814956
      43099.71c -5.86869689244e-05
      42095.71c -5.18634996991e-05
      61147.71c -2.1589456095e-05
      40090.71c -0.906182486425
      45103.71c -2.8669623396e-05
      55133.71c -8.60653517968e-05
      93237.71c -2.82931326136e-06
      55135.71c -5.8997504998e-05
      94240.71c -1.96055269066e-05
      94241.71c -2.09909440413e-06
      94242.71c -8.30330736589e-08
m6094 62152.71c -8.5254422246e-06
      62151.71c -2.13339619927e-06
      62150.71c -1.62158420948e-05
      64155.71c -8.49256701989e-09
      63153.71c -3.00735205003e-06
      60143.71c -8.18221052307e-05
      60145.71c -5.81849315452e-05
      95241.71c -8.10153289656e-08
      95243.71c -1.00490803268e-09
      94239.71c -0.000212210466348
      94238.71c -8.36426031489e-08
      63154.71c -1.38537425046e-07
      44103.71c -1.86771725175e-06
      92238.71c -0.0625036676937
      54135.71c -3.18658835478e-08
      62149.71c -5.86669439427e-07
      54131.71c -3.79253146789e-05
      1001.71c -0.0171340556346
      62147.71c -9.27187895125e-06
      92234.71c -6.83744432437e-08
      92235.71c -0.0128556576214
      92236.71c -0.000455715986343
      43099.71c -6.17904738534e-05
      42095.71c -5.46437299851e-05
      61147.71c -2.26634153675e-05
      40090.71c -0.906270659834
      45103.71c -3.01771219821e-05
      55133.71c -9.06091919515e-05
      93237.71c -3.11766041232e-06
      55135.71c -6.10133715975e-05
      94240.71c -2.15392954624e-05
      94241.71c -2.42311402393e-06
      94242.71c -1.01806033626e-07
c **  F-11  Fuel ID= 5949
m6111 62152.71c -6.88435750897e-06
      62151.71c -2.03933480298e-06
      62150.71c -1.33754945677e-05
      64155.71c -1.1035568493e-08
      63153.71c -2.38759613565e-06
      60143.71c -6.92265036979e-05
      60145.71c -4.87120302819e-05
      95241.71c -4.64117117918e-08
      95243.71c -2.61137616088e-10
      94239.71c -0.000160141975459
      94238.71c -4.01517652176e-08
      63154.71c -8.93089786373e-08
      44103.71c -1.22851027906e-06
      92238.71c -0.0625649965473
      54135.71c -2.40925198314e-08
      62149.71c -5.84260707225e-07
      54131.71c -3.18639154612e-05
      1001.71c -0.017128492115
      62147.71c -9.48768931809e-06
      92234.71c -5.19821645321e-08
      92235.71c -0.0133152053075
      92236.71c -0.000376764837014
      43099.71c -5.16947900141e-05
      42095.71c -4.71707569967e-05
      61147.71c -1.76389416104e-05
      40090.71c -0.9059763889
      45103.71c -2.55181400198e-05
      55133.71c -7.59079169238e-05
      93237.71c -1.89477825792e-06
      55135.71c -5.76270531824e-05
      94240.71c -1.33736477379e-05
      94241.71c -1.09419638657e-06
      94242.71c -3.71602262069e-08
m6112 62152.71c -8.89695932915e-06
      62151.71c -2.13837851968e-06
      62150.71c -1.68026102984e-05
      64155.71c -1.02774641781e-08
      63153.71c -3.144059402e-06
      60143.71c -8.50098262848e-05
      60145.71c -6.028886708e-05
      95241.71c -1.12219939911e-07
      95243.71c -1.18823136908e-09
      94239.71c -0.000217300684507
      94238.71c -9.38453165385e-08
      63154.71c -1.46295918148e-07
      44103.71c -1.52675189565e-06
      92238.71c -0.0624976523891
      54135.71c -2.77506775536e-08
      62149.71c -5.79932117648e-07
      54131.71c -3.93794202987e-05
      1001.71c -0.0171351215497
      62147.71c -1.16734222248e-05
      92234.71c -7.14098505045e-08
      92235.71c -0.0127546783028
      92236.71c -0.000471736413523
      43099.71c -6.40886867173e-05
      42095.71c -5.85765541767e-05
      61147.71c -2.15498103825e-05
      40090.71c -0.906327039222
      45103.71c -3.16418079274e-05
      55133.71c -9.40714853478e-05
      93237.71c -3.33438752975e-06
      55135.71c -6.76454104825e-05
      94240.71c -2.2916011537e-05
      94241.71c -2.62856699927e-06
      94242.71c -1.15502580404e-07
m6113 62152.71c -1.08069335803e-05
      62151.71c -2.16541416934e-06
      62150.71c -1.99859955083e-05
      64155.71c -9.62745524547e-09
      63153.71c -3.86382860047e-06
      60143.71c -9.9262770648e-05
      60145.71c -7.09668527859e-05
      95241.71c -1.79203090333e-07
      95243.71c -2.76495757975e-09
      94239.71c -0.000245781148811
      94238.71c -1.53690215717e-07
      63154.71c -2.04225084771e-07
      44103.71c -1.78910930459e-06
      92238.71c -0.0624619429065
      54135.71c -3.0346724323e-08
      62149.71c -5.65713448072e-07
      54131.71c -4.62867434644e-05
      1001.71c -0.0171412848841
      62147.71c -1.37071471404e-05
      92234.71c -8.26880779566e-08
      92235.71c -0.0122403318218
      92236.71c -0.000554182986932
      43099.71c -7.55389845886e-05
      42095.71c -6.91729237569e-05
      61147.71c -2.51145894043e-05
      40090.71c -0.906653036131
      45103.71c -3.72359672393e-05
      55133.71c -0.000110845332374
      93237.71c -4.57380591323e-06
      55135.71c -7.5554716613e-05
      94240.71c -3.09401815177e-05
      94241.71c -4.17652564022e-06
      94242.71c -2.24039926506e-07
m6114 62152.71c -1.14576805336e-05
      62151.71c -2.16856232981e-06
      62150.71c -2.10719963506e-05
      64155.71c -9.43837396868e-09
      63153.71c -4.11911073001e-06
      60143.71c -0.000104026327972
      60145.71c -7.45837716255e-05
      95241.71c -2.0573287512e-07
      95243.71c -3.55698088044e-09
      94239.71c -0.000254527596218
      94238.71c -1.78581380358e-07
      63154.71c -2.26008709705e-07
      44103.71c -1.87690089117e-06
      92238.71c -0.0624498536402
      54135.71c -3.11021523363e-08
      62149.71c -5.60809803771e-07
      54131.71c -4.86205849529e-05
      1001.71c -0.0171433893863
      62147.71c -1.43949126433e-05
      92234.71c -8.64243133459e-08
      92235.71c -0.0120662108766
      92236.71c -0.000581940489874
      43099.71c -7.94241935531e-05
      42095.71c -7.277885902e-05
      61147.71c -2.63033448071e-05
      40090.71c -0.906764349447
      45103.71c -3.91279763899e-05
      55133.71c -0.000116533712316
      93237.71c -5.03388782019e-06
      55135.71c -7.80164828678e-05
      94240.71c -3.38283326003e-05
      94241.71c -4.78701330456e-06
      94242.71c -2.73257967841e-07
c **  F-12  Fuel ID= 2900
m6121 62152.71c -7.50774480455e-07
      62151.71c -6.82006194621e-07
      62150.71c -1.63213895992e-06
      64155.71c -1.90702743537e-09
      63153.71c -3.39096347746e-07
      60143.71c -1.05103399217e-05
      60145.71c -7.86019826642e-06
      95241.71c -1.62957644335e-10
      95243.71c -2.25397891681e-13
      94239.71c -2.97445982041e-05
      94238.71c -5.55430521687e-10
      63154.71c -3.48099114349e-09
      44103.71c -1.05041590749e-06
      92238.71c -0.0626565116448
      54135.71c -2.3852878235e-08
      62149.71c -5.78415145324e-07
      54131.71c -4.92957743285e-06
      1001.71c -0.017106041717
      62147.71c -4.08930829917e-07
      92234.71c -8.79244819778e-09
      92235.71c -0.015287316174
      92236.71c -6.13891459516e-05
      43099.71c -8.18287047549e-06
      42095.71c -4.22920925124e-06
      61147.71c -3.7851326425e-06
      40090.71c -0.904788921238
      45103.71c -3.27800716515e-06
      55133.71c -1.18072989379e-05
      93237.71c -1.00440886363e-07
      55135.71c -9.37012259572e-06
      94240.71c -5.29193364964e-07
      94241.71c -1.24496024189e-08
      94242.71c -1.11888652177e-10
m6122 62152.71c -9.98112221886e-07
      62151.71c -8.18635637384e-07
      62150.71c -2.19326615284e-06
      64155.71c -2.0776340335e-09
      63153.71c -4.34592691064e-07
      60143.71c -1.32870907396e-05
      60145.71c -9.95346601556e-06
      95241.71c -4.33781244192e-10
      95243.71c -1.13800892115e-12
      94239.71c -4.28144987342e-05
      94238.71c -1.23959704446e-09
      63154.71c -5.87333208226e-09
      44103.71c -1.33402058517e-06
      92238.71c -0.0626459542957
      54135.71c -2.85487668131e-08
      62149.71c -6.1131116079e-07
      54131.71c -6.24184482752e-06
      1001.71c -0.0171072266651
      62147.71c -5.16441915476e-07
      92234.71c -1.23741430945e-08
      92235.71c -0.0151850119172
      92236.71c -7.89292459924e-05
      43099.71c -1.03667807385e-05
      42095.71c -5.35996988707e-06
      61147.71c -4.77922736856e-06
      40090.71c -0.904851596629
      45103.71c -4.1552708412e-06
      55133.71c -1.49569814325e-05
      93237.71c -1.68843788155e-07
      55135.71c -1.12285495931e-05
      94240.71c -9.78299147629e-07
      94241.71c -3.31129711844e-08
      94242.71c -3.823240114e-10
m6123 62152.71c -1.24462483133e-06
      62151.71c -9.31079220226e-07
      62150.71c -2.73545822182e-06
      64155.71c -2.17422413331e-09
      63153.71c -5.24535664105e-07
      60143.71c -1.58772118691e-05
      60145.71c -1.19122039053e-05
      95241.71c -7.40711434496e-10
      95243.71c -2.82619660061e-12
      94239.71c -5.09159935754e-05
      94238.71c -1.99118628267e-09
      63154.71c -8.39289234838e-09
      44103.71c -1.59821068503e-06
      92238.71c -0.0626403315812
      54135.71c -3.23546454935e-08
      62149.71c -6.25388799509e-07
      54131.71c -7.46868609946e-06
      1001.71c -0.0171083414338
      62147.71c -6.17056102837e-07
      92234.71c -1.47311098005e-08
      92235.71c -0.0150899741716
      92236.71c -9.44365828353e-05
      43099.71c -1.24111091277e-05
      42095.71c -6.42066483405e-06
      61147.71c -5.7077934854e-06
      40090.71c -0.904910560029
      45103.71c -4.97461161471e-06
      55133.71c -1.79050741817e-05
      93237.71c -2.27877771968e-07
      55135.71c -1.27488147822e-05
      94240.71c -1.39209386651e-06
      94241.71c -5.653776177e-08
      94242.71c -7.87516879145e-10
m6124 62152.71c -1.33296796924e-06
      62151.71c -9.67206628695e-07
      62150.71c -2.92552806002e-06
      64155.71c -2.19773455847e-09
      63153.71c -5.56126296151e-07
      60143.71c -1.67721465806e-05
      60145.71c -1.25915091581e-05
      95241.71c -8.69353086209e-10
      95243.71c -3.71110079322e-12
      94239.71c -5.35736536829e-05
      94238.71c -2.30178437476e-09
      63154.71c -9.36864881264e-09
      44103.71c -1.69049574154e-06
      92238.71c -0.0626385125262
      54135.71c -3.35926342732e-08
      62149.71c -6.28561853088e-07
      54131.71c -7.89360520919e-06
      1001.71c -0.0171087286822
      62147.71c -6.51765720157e-07
      92234.71c -1.54787248916e-08
      92235.71c -0.0150570371726
      92236.71c -9.98054939863e-05
      43099.71c -1.31201822632e-05
      42095.71c -6.78812927663e-06
      61147.71c -6.02865114208e-06
      40090.71c -0.904931042737
      45103.71c -5.25825251455e-06
      55133.71c -1.89271705065e-05
      93237.71c -2.4985759084e-07
      55135.71c -1.32381974671e-05
      94240.71c -1.54824155203e-06
      94241.71c -6.63465841458e-08
      94242.71c -9.79922091889e-10
c **  F-13  Fuel ID= 3329
m6131 62152.71c -3.60183963259e-06
      62151.71c -1.56156512973e-06
      62150.71c -7.44972955835e-06
      64155.71c -7.25507242654e-09
      63153.71c -1.35243078703e-06
      60143.71c -3.98847141005e-05
      60145.71c -2.82418898056e-05
      95241.71c -3.34220873541e-08
      95243.71c -3.25582693707e-10
      94239.71c -9.55206833126e-05
      94238.71c -2.29800244776e-08
      63154.71c -4.2342446536e-08
      44103.71c -1.17885038376e-06
      92238.71c -0.0626121706184
      54135.71c -2.38932280614e-08
      62149.71c -6.15390308322e-07
      54131.71c -1.83706289783e-05
      1001.71c -0.0171172388864
      62147.71c -4.38121027834e-06
      92234.71c -3.01561254528e-08
      92235.71c -0.0143030323686
      92236.71c -0.00021869493899
      43099.71c -2.98814492234e-05
      42095.71c -2.50056339336e-05
      61147.71c -1.12916987082e-05
      40090.71c -0.905381172498
      45103.71c -1.43274072759e-05
      55133.71c -4.37764207312e-05
      93237.71c -9.29551730905e-07
      55135.71c -3.34066692112e-05
      94240.71c -6.16428847196e-06
      94241.71c -5.59611497534e-07
      94242.71c -2.86523222233e-08
m6132 62152.71c -4.63197122791e-06
      62151.71c -1.73442301478e-06
      62150.71c -9.37869445173e-06
      64155.71c -7.22081451605e-09
      63153.71c -1.75173090455e-06
      60143.71c -4.88995711727e-05
      60145.71c -3.48407350808e-05
      95241.71c -7.52765798009e-08
      95243.71c -1.36486309959e-09
      94239.71c -0.000131627171282
      94238.71c -5.27688341791e-08
      63154.71c -6.84914422712e-08
      44103.71c -1.4668000847e-06
      92238.71c -0.0625740355445
      54135.71c -2.78922725731e-08
      62149.71c -6.1790635158e-07
      54131.71c -2.26347728091e-05
      1001.71c -0.0171210008166
      62147.71c -5.34338226706e-06
      92234.71c -4.16488150663e-08
      92235.71c -0.0139824783208
      92236.71c -0.000273268443188
      43099.71c -3.69105708732e-05
      42095.71c -3.09009287822e-05
      61147.71c -1.38321879129e-05
      40090.71c -0.905580152066
      45103.71c -1.76944391961e-05
      55133.71c -5.40565711569e-05
      93237.71c -1.60372663236e-06
      55135.71c -3.914930889e-05
      94240.71c -1.03604773748e-05
      94241.71c -1.27188596392e-06
      94242.71c -8.28903597672e-08
m6133 62152.71c -5.65518841011e-06
      62151.71c -1.85146140041e-06
      62150.71c -1.12249936578e-05
      64155.71c -7.06767840732e-09
      63153.71c -2.13566623094e-06
      60143.71c -5.73352669216e-05
      60145.71c -4.11118583092e-05
      95241.71c -1.14691798658e-07
      95243.71c -3.02447514911e-09
      94239.71c -0.000151373228289
      94238.71c -8.5739876135e-08
      63154.71c -9.51252676637e-08
      44103.71c -1.73575098623e-06
      92238.71c -0.0625539214589
      54135.71c -3.10720374e-08
      62149.71c -6.10008722322e-07
      54131.71c -2.6674780629e-05
      1001.71c -0.0171245981296
      62147.71c -6.25631565455e-06
      92234.71c -4.8656772873e-08
      92235.71c -0.0136796075868
      92236.71c -0.000322019329432
      43099.71c -4.35976393169e-05
      42095.71c -3.65266874698e-05
      61147.71c -1.62420790582e-05
      40090.71c -0.905770424541
      45103.71c -2.08650593002e-05
      55133.71c -6.38316526265e-05
      93237.71c -2.18123278054e-06
      55135.71c -4.38546946561e-05
      94240.71c -1.38706825079e-05
      94241.71c -1.95528924473e-06
      94242.71c -1.54040126162e-07
m6134 62152.71c -6.00784031394e-06
      62151.71c -1.88443926691e-06
      62150.71c -1.18549574072e-05
      64155.71c -7.00645539777e-09
      63153.71c -2.27116818403e-06
      60143.71c -6.01680243832e-05
      60145.71c -4.3239776243e-05
      95241.71c -1.29549667851e-07
      95243.71c -3.83758499332e-09
      94239.71c -0.000157619797695
      94238.71c -9.93982303549e-08
      63154.71c -1.05104600651e-07
      44103.71c -1.82845273928e-06
      92238.71c -0.0625472059367
      54135.71c -3.2072930278e-08
      62149.71c -6.0718423113e-07
      54131.71c -2.80433262818e-05
      1001.71c -0.0171258258501
      62147.71c -6.56282098311e-06
      92234.71c -5.09401178017e-08
      92235.71c -0.0135768879839
      92236.71c -0.000338487424607
      43099.71c -4.58691562877e-05
      42095.71c -3.84375658842e-05
      61147.71c -1.7053809985e-05
      40090.71c -0.905835362295
      45103.71c -2.19380517962e-05
      55133.71c -6.71517069345e-05
      93237.71c -2.3987006165e-06
      55135.71c -4.53274611632e-05
      94240.71c -1.51365678746e-05
      94241.71c -2.21656941775e-06
      94242.71c -1.85222778094e-07
c **  F-14  Fuel ID= 5653
m6141 62152.71c -6.86123024431e-06
      62151.71c -2.03827574162e-06
      62150.71c -1.33370201132e-05
      64155.71c -1.10237196362e-08
      63153.71c -2.37995324395e-06
      60143.71c -6.90426124306e-05
      60145.71c -4.85796334529e-05
      95241.71c -4.5935521169e-08
      95243.71c -2.57537268046e-10
      94239.71c -0.000159789137657
      94238.71c -3.98303321983e-08
      63154.71c -8.88396495102e-08
      44103.71c -1.22815209954e-06
      92238.71c -0.0625653203942
      54135.71c -2.40785113877e-08
      62149.71c -5.84496854732e-07
      54131.71c -3.17772464795e-05
      1001.71c -0.0171284181616
      62147.71c -9.4434997874e-06
      92234.71c -5.18471076939e-08
      92235.71c -0.0133215897507
      92236.71c -0.000375750212367
      43099.71c -5.15532816846e-05
      42095.71c -4.70265413132e-05
      61147.71c -1.76103546369e-05
      40090.71c -0.905972477288
      45103.71c -2.54461318287e-05
      55133.71c -7.56998328124e-05
      93237.71c -1.88509367885e-06
      55135.71c -5.7472127796e-05
      94240.71c -1.33051856364e-05
      94241.71c -1.08581553008e-06
      94242.71c -3.67574854801e-08
m6142 62152.71c -8.86838478183e-06
      62151.71c -2.1377945934e-06
      62150.71c -1.67554167517e-05
      64155.71c -1.02672811874e-08
      63153.71c -3.1336413944e-06
      60143.71c -8.47905721496e-05
      60145.71c -6.01281383964e-05
      95241.71c -1.11101530634e-07
      95243.71c -1.17206934678e-09
      94239.71c -0.000216862790137
      94238.71c -9.30888507764e-08
      63154.71c -1.45531774003e-07
      44103.71c -1.52636702742e-06
      92238.71c -0.0624981586919
      54135.71c -2.77376200457e-08
      62149.71c -5.80219128265e-07
      54131.71c -3.92746579882e-05
      1001.71c -0.017135030801
      62147.71c -1.16198329847e-05
      92234.71c -7.12292649637e-08
      92235.71c -0.0127624191982
      92236.71c -0.000470496340123
      43099.71c -6.39163238268e-05
      42095.71c -5.84001960641e-05
      61147.71c -2.15167986886e-05
      40090.71c -0.906322239256
      45103.71c -3.15542646345e-05
      55133.71c -9.38182592959e-05
      93237.71c -3.3174106088e-06
      55135.71c -6.74670866666e-05
      94240.71c -2.28038817972e-05
      94241.71c -2.60926829889e-06
      94242.71c -1.14279059796e-07
m6143 62152.71c -1.0773719536e-05
      62151.71c -2.16523598083e-06
      62150.71c -1.99311579567e-05
      64155.71c -9.61733920458e-09
      63153.71c -3.85085707926e-06
      60143.71c -9.90147147028e-05
      60145.71c -7.0781683425e-05
      95241.71c -1.77460587919e-07
      95243.71c -2.72821313012e-09
      94239.71c -0.000245327867989
      94238.71c -1.52451132901e-07
      63154.71c -2.03173041965e-07
      44103.71c -1.78893343962e-06
      92238.71c -0.0624625569438
      54135.71c -3.03423489475e-08
      62149.71c -5.66047966848e-07
      54131.71c -4.61663837534e-05
      1001.71c -0.0171411790434
      62147.71c -1.36450180136e-05
      92234.71c -8.24842739411e-08
      92235.71c -0.012249248276
      92236.71c -0.000552760239593
      43099.71c -7.53397909829e-05
      42095.71c -6.89679603029e-05
      61147.71c -2.50782824662e-05
      40090.71c -0.906647437904
      45103.71c -3.71351443512e-05
      55133.71c -0.00011055278513
      93237.71c -4.550695057e-06
      55135.71c -7.53598460032e-05
      94240.71c -3.07945183942e-05
      94241.71c -4.14697853038e-06
      94242.71c -2.21715421038e-07
m6144 62152.71c -1.14229847321e-05
      62151.71c -2.16849151343e-06
      62150.71c -2.10146255895e-05
      64155.71c -9.4278441468e-09
      63153.71c -4.10524568805e-06
      60143.71c -0.000103769288845
      60145.71c -7.43906430092e-05
      95241.71c -2.0374915597e-07
      95243.71c -3.50989508964e-09
      94239.71c -0.000254071153097
      94238.71c -1.77142614758e-07
      63154.71c -2.24850131925e-07
      44103.71c -1.87682926469e-06
      92238.71c -0.0624505293988
      54135.71c -3.11009236962e-08
      62149.71c -5.61172863353e-07
      54131.71c -4.84952085194e-05
      1001.71c -0.017143278167
      62147.71c -1.43299593977e-05
      92234.71c -8.6214353846e-08
      92235.71c -0.0120755067667
      92236.71c -0.000580459113868
      43099.71c -7.92162251007e-05
      42095.71c -7.25643671664e-05
      61147.71c -2.62661260557e-05
      40090.71c -0.906758466733
      45103.71c -3.90228204663e-05
      55133.71c -0.000116228329524
      93237.71c -5.0085024759e-06
      55135.71c -7.78168195135e-05
      94240.71c -3.36709762015e-05
      94241.71c -4.75361463242e-06
      94242.71c -2.70442052259e-07
c **  F-15  Fuel ID= 2448
m6151 62152.71c -2.85210821904e-06
      62151.71c -1.51878785184e-06
      62150.71c -6.18092509911e-06
      64155.71c -6.74356986704e-09
      63153.71c -1.08238638405e-06
      60143.71c -3.39051860213e-05
      60145.71c -2.38780444531e-05
      95241.71c -4.50299097255e-09
      95243.71c -1.37316062232e-11
      94239.71c -8.43007458269e-05
      94238.71c -6.49541695707e-09
      63154.71c -2.43913870547e-08
      44103.71c -1.17388703215e-06
      92238.71c -0.0626234563138
      54135.71c -2.38925715783e-08
      62149.71c -6.21838421791e-07
      54131.71c -1.55188767878e-05
      1001.71c -0.0171147895857
      62147.71c -2.80494214691e-06
      92234.71c -2.56642179113e-08
      92235.71c -0.0145135222082
      92236.71c -0.000185249413276
      43099.71c -2.52044233803e-05
      42095.71c -2.02237273767e-05
      61147.71c -1.04814241239e-05
      40090.71c -0.905251621773
      45103.71c -1.19578862306e-05
      55133.71c -3.68967713642e-05
      93237.71c -5.46574020737e-07
      55135.71c -2.82851966647e-05
      94240.71c -3.65720572141e-06
      94241.71c -1.74729059044e-07
      94242.71c -3.33596391475e-09
m6152 62152.71c -3.70537376725e-06
      62151.71c -1.69227599144e-06
      62150.71c -7.79426114948e-06
      64155.71c -6.69763504916e-09
      63153.71c -1.37660967839e-06
      60143.71c -4.16681247717e-05
      60145.71c -2.94522399594e-05
      95241.71c -1.1054723905e-08
      95243.71c -6.17672682215e-11
      94239.71c -0.000117635894989
      94238.71c -1.45018311931e-08
      63154.71c -3.92942566903e-08
      44103.71c -1.4540963137e-06
      92238.71c -0.0625920158532
      54135.71c -2.78594053706e-08
      62149.71c -6.26379917677e-07
      54131.71c -1.91328776541e-05
      1001.71c -0.0171179312155
      62147.71c -3.44598052969e-06
      92234.71c -3.54261398954e-08
      92235.71c -0.0142421457747
      92236.71c -0.000231757726706
      43099.71c -3.11172754317e-05
      42095.71c -2.49953784355e-05
      61147.71c -1.28461027023e-05
      40090.71c -0.905417791826
      45103.71c -1.47753575644e-05
      55133.71c -4.55445046053e-05
      93237.71c -9.37814414088e-07
      55135.71c -3.31861203619e-05
      94240.71c -6.39725135756e-06
      94241.71c -4.28457433769e-07
      94242.71c -1.03306366217e-08
m6153 62152.71c -4.5867531081e-06
      62151.71c -1.81308549634e-06
      62150.71c -9.38620926119e-06
      64155.71c -6.52274803365e-09
      63153.71c -1.66856913843e-06
      60143.71c -4.91974149006e-05
      60145.71c -3.49088160783e-05
      95241.71c -1.8192013571e-08
      95243.71c -1.47357678814e-10
      94239.71c -0.000136660787866
      94238.71c -2.35009468544e-08
      63154.71c -5.52178646611e-08
      44103.71c -1.72357453775e-06
      92238.71c -0.0625757707707
      54135.71c -3.11373097102e-08
      62149.71c -6.20042142736e-07
      54131.71c -2.26638242208e-05
      1001.71c -0.0171210216311
      62147.71c -4.07579277625e-06
      92234.71c -4.16912199613e-08
      92235.71c -0.0139785376259
      92236.71c -0.000274440176429
      43099.71c -3.69086983017e-05
      42095.71c -2.96858019891e-05
      61147.71c -1.51545996283e-05
      40090.71c -0.905581253004
      45103.71c -1.75187911898e-05
      55133.71c -5.40128592412e-05
      93237.71c -1.28277273442e-06
      55135.71c -3.73582161703e-05
      94240.71c -8.84919817873e-06
      94241.71c -7.04096647677e-07
      94242.71c -2.04784714398e-08
m6154 62152.71c -4.8941810976e-06
      62151.71c -1.84717136958e-06
      62150.71c -9.93095834141e-06
      64155.71c -6.45218826716e-09
      63153.71c -1.77077950527e-06
      60143.71c -5.17484113258e-05
      60145.71c -3.67681619341e-05
      95241.71c -2.11171128098e-08
      95243.71c -1.91092611653e-10
      94239.71c -0.000142946388818
      94238.71c -2.71907329748e-08
      63154.71c -6.12500653785e-08
      44103.71c -1.81537091448e-06
      92238.71c -0.0625701729412
      54135.71c -3.21569044459e-08
      62149.71c -6.17866667994e-07
      54131.71c -2.38657968957e-05
      1001.71c -0.0171220795405
      62147.71c -4.28979269621e-06
      92234.71c -4.37224540133e-08
      92235.71c -0.0138887136859
      92236.71c -0.000288982259382
      43099.71c -3.88840905072e-05
      42095.71c -3.12882503088e-05
      61147.71c -1.59364692876e-05
      40090.71c -0.905637208951
      45103.71c -1.8453675905e-05
      55133.71c -5.69006492656e-05
      93237.71c -1.40842146699e-06
      55135.71c -3.86772962363e-05
      94240.71c -9.76472767512e-06
      94241.71c -8.16890399949e-07
      94242.71c -2.51913072339e-08
c **  F-16  Fuel ID= 5655
m6161 62152.71c -6.9751349386e-06
      62151.71c -2.04328534172e-06
      62150.71c -1.35262613648e-05
      64155.71c -1.10810372526e-08
      63153.71c -2.41772216875e-06
      60143.71c -6.99467381423e-05
      60145.71c -4.92307221523e-05
      95241.71c -4.83633203211e-08
      95243.71c -2.76292911755e-10
      94239.71c -0.000161513441981
      94238.71c -4.14558654132e-08
      63154.71c -9.11758461583e-08
      44103.71c -1.22976478524e-06
      92238.71c -0.0625637310652
      54135.71c -2.41235598381e-08
      62149.71c -5.83371262552e-07
      54131.71c -3.22036477519e-05
      1001.71c -0.0171287817875
      62147.71c -9.6622900923e-06
      92234.71c -5.25122675712e-08
      92235.71c -0.0132901981242
      92236.71c -0.000380738772372
      43099.71c -5.22494953714e-05
      42095.71c -4.77358777786e-05
      61147.71c -1.77495145158e-05
      40090.71c -0.905991710533
      45103.71c -2.58002973483e-05
      55133.71c -7.67238563191e-05
      93237.71c -1.93335696488e-06
      55135.71c -5.82337234663e-05
      94240.71c -1.36454250984e-05
      94241.71c -1.1279834178e-06
      94242.71c -3.88196151458e-08
m6162 62152.71c -9.00891900677e-06
      62151.71c -2.14049674627e-06
      62150.71c -1.69874145259e-05
      64155.71c -1.03171892657e-08
      63153.71c -3.18513504553e-06
      60143.71c -8.58674966191e-05
      60145.71c -6.09179663513e-05
      95241.71c -1.16791892803e-07
      95243.71c -1.25587012159e-09
      94239.71c -0.000219001963897
      94238.71c -9.69136144304e-08
      63154.71c -1.49333145431e-07
      44103.71c -1.52800550628e-06
      92238.71c -0.0624956478227
      54135.71c -2.77738462943e-08
      62149.71c -5.78866700599e-07
      54131.71c -3.978972915e-05
      1001.71c -0.0171354771915
      62147.71c -1.18850090474e-05
      92234.71c -7.21160611848e-08
      92235.71c -0.0127243832811
      92236.71c -0.00047658794169
      43099.71c -6.47637628242e-05
      42095.71c -5.92671713298e-05
      61147.71c -2.16771919006e-05
      40090.71c -0.906345850165
      45103.71c -3.1984597441e-05
      55133.71c -9.50636293321e-05
      93237.71c -3.40197822869e-06
      55135.71c -6.83431189975e-05
      94240.71c -2.33599125753e-05
      94241.71c -2.70620035905e-06
      94242.71c -1.20531274908e-07
m6163 62152.71c -1.09369042531e-05
      62151.71c -2.16599241906e-06
      62150.71c -2.02005745361e-05
      64155.71c -9.66728499128e-09
      63153.71c -3.91496222476e-06
      60143.71c -0.00010023180978
      60145.71c -7.16911865367e-05
      95241.71c -1.86310213797e-07
      95243.71c -2.91890061037e-09
      94239.71c -0.000247540211609
      94238.71c -1.58715378589e-07
      63154.71c -2.08406081172e-07
      44103.71c -1.78962218841e-06
      92238.71c -0.0624595265037
      54135.71c -3.03422634198e-08
      62149.71c -5.64463736526e-07
      54131.71c -4.67576549827e-05
      1001.71c -0.0171416991171
      62147.71c -1.39522914068e-05
      92234.71c -8.34863223935e-08
      92235.71c -0.0122054614859
      92236.71c -0.000559745642574
      43099.71c -7.6318618789e-05
      42095.71c -6.99749655189e-05
      61147.71c -2.52543460045e-05
      40090.71c -0.906674946132
      45103.71c -3.76303300527e-05
      55133.71c -0.000111990605997
      93237.71c -4.66560761349e-06
      55135.71c -7.63166138641e-05
      94240.71c -3.15158123579e-05
      94241.71c -4.29512122726e-06
      94242.71c -2.33577068362e-07
m6164 62152.71c -1.15933260025e-05
      62151.71c -2.16872351709e-06
      62150.71c -2.12963438734e-05
      64155.71c -9.47994824809e-09
      63153.71c -4.17376046328e-06
      60143.71c -0.000105029650287
      60145.71c -7.53387838109e-05
      95241.71c -2.13817755477e-07
      95243.71c -3.75389691982e-09
      94239.71c -0.000256291134962
      94238.71c -1.84411629337e-07
      63154.71c -2.30610115866e-07
      44103.71c -1.87697039927e-06
      92238.71c -0.0624472412306
      54135.71c -3.1086629427e-08
      62149.71c -5.59467256447e-07
      54131.71c -4.91108130796e-05
      1001.71c -0.017143823781
      62147.71c -1.465113773e-05
      92234.71c -8.72485479319e-08
      92235.71c -0.0120298737254
      92236.71c -0.000587728889495
      43099.71c -8.02376735574e-05
      42095.71c -7.36178508882e-05
      61147.71c -2.64463676057e-05
      40090.71c -0.906787325864
      45103.71c -3.95390273438e-05
      55133.71c -0.00011772850031
      93237.71c -5.13472821833e-06
      55135.71c -7.87967532254e-05
      94240.71c -3.44492858747e-05
      94241.71c -4.92100386722e-06
      94242.71c -2.8479837965e-07
c **  F-17  Fuel ID= 5939
m6171 62152.71c -7.2485263588e-06
      62151.71c -2.05401473852e-06
      62150.71c -1.39789041419e-05
      64155.71c -1.12120903372e-08
      63153.71c -2.50916900862e-06
      60143.71c -7.21070037711e-05
      60145.71c -5.07872891401e-05
      95241.71c -5.47078787569e-08
      95243.71c -3.27728630542e-10
      94239.71c -0.000165567621637
      94238.71c -4.56235695721e-08
      63154.71c -9.69379874531e-08
      44103.71c -1.23268603241e-06
      92238.71c -0.0625599518789
      54135.71c -2.40815106578e-08
      62149.71c -5.8090460272e-07
      54131.71c -3.32242364889e-05
      1001.71c -0.0171296507673
      62147.71c -1.01946460259e-05
      92234.71c -5.41086932164e-08
      92235.71c -0.0132151823119
      92236.71c -0.00039265829879
      43099.71c -5.39159363294e-05
      42095.71c -4.94324906519e-05
      61147.71c -1.80732714394e-05
      40090.71c -0.906037673432
      45103.71c -2.66473224215e-05
      55133.71c -7.91765701515e-05
      93237.71c -2.05274407686e-06
      55135.71c -6.00527106855e-05
      94240.71c -1.44812197002e-05
      94241.71c -1.23478253476e-06
      94242.71c -4.42619205532e-08
m6172 62152.71c -9.34497781337e-06
      62151.71c -2.1458926483e-06
      62150.71c -1.75415133372e-05
      64155.71c -1.04354444818e-08
      63153.71c -3.30988070454e-06
      60143.71c -8.8433931354e-05
      60145.71c -6.28025229525e-05
      95241.71c -1.31591552417e-07
      95243.71c -1.48335250731e-09
      94239.71c -0.000224026535618
      94238.71c -1.06715081407e-07
      63154.71c -1.58694585603e-07
      44103.71c -1.53032605135e-06
      92238.71c -0.0624895130185
      54135.71c -2.76846657514e-08
      62149.71c -5.76004055759e-07
      54131.71c -4.10203690295e-05
      1001.71c -0.0171365451065
      62147.71c -1.25292697503e-05
      92234.71c -7.42281579821e-08
      92235.71c -0.0126336483278
      92236.71c -0.000491109042385
      43099.71c -6.67885472659e-05
      42095.71c -6.13379638272e-05
      61147.71c -2.20484069138e-05
      40090.71c -0.906402335334
      45103.71c -3.30122734669e-05
      55133.71c -9.80414476214e-05
      93237.71c -3.61093951155e-06
      55135.71c -7.0431734857e-05
      94240.71c -2.47181769695e-05
      94241.71c -2.9506672328e-06
      94242.71c -1.36956476038e-07
m6173 62152.71c -1.13260763366e-05
      62151.71c -2.16704811817e-06
      62150.71c -2.08430496332e-05
      64155.71c -9.78793518867e-09
      63153.71c -4.07019794812e-06
      60143.71c -0.000103124130295
      60145.71c -7.38586054612e-05
      95241.71c -2.09229054334e-07
      95243.71c -3.43752212441e-09
      94239.71c -0.000252724032754
      94238.71c -1.74764212534e-07
      63154.71c -2.2128989036e-07
      44103.71c -1.79016266881e-06
      92238.71c -0.0624522138441
      54135.71c -3.02063727707e-08
      62149.71c -5.61058402819e-07
      54131.71c -4.81673057654e-05
      1001.71c -0.0171429398276
      62147.71c -1.46978271552e-05
      92234.71c -8.58806788384e-08
      92235.71c -0.0121011674021
      92236.71c -0.000576375080154
      43099.71c -7.86540192337e-05
      42095.71c -7.23764926692e-05
      61147.71c -2.56596656896e-05
      40090.71c -0.906740570965
      45103.71c -3.88102101795e-05
      55133.71c -0.000115422745416
      93237.71c -4.94824584736e-06
      55135.71c -7.85942458817e-05
      94240.71c -3.3271408973e-05
      94241.71c -4.66711815464e-06
      94242.71c -2.64638335412e-07
m6174 62152.71c -1.19987889775e-05
      62151.71c -2.16854799188e-06
      62150.71c -2.19672497665e-05
      64155.71c -9.60653382976e-09
      63153.71c -4.33963606739e-06
      60143.71c -0.000108019690733
      60145.71c -7.75954113221e-05
      95241.71c -2.39856460147e-07
      95243.71c -4.41542094596e-09
      94239.71c -0.000261443514671
      94238.71c -2.03005709688e-07
      63154.71c -2.44772775205e-07
      44103.71c -1.87598185284e-06
      92238.71c -0.0624396000856
      54135.71c -3.09247214579e-08
      62149.71c -5.55890952946e-07
      54131.71c -5.05765184354e-05
      1001.71c -0.0171451198364
      62147.71c -1.54300214396e-05
      92234.71c -8.97315309939e-08
      92235.71c -0.0119212895373
      92236.71c -0.000605013396157
      43099.71c -8.26716296061e-05
      42095.71c -7.61281120485e-05
      61147.71c -2.68597535936e-05
      40090.71c -0.90685587805
      45103.71c -4.07673577886e-05
      55133.71c -0.000121304854483
      93237.71c -5.44521653559e-06
      55135.71c -8.11272340156e-05
      94240.71c -3.63381231808e-05
      94241.71c -5.34093380852e-06
      94242.71c -3.22314075254e-07
c **  F-18  Fuel ID= 5946
m6181 62152.71c -6.99835887949e-06
      62151.71c -2.04419677806e-06
      62150.71c -1.35647122755e-05
      64155.71c -1.10921698956e-08
      63153.71c -2.42549035695e-06
      60143.71c -7.01302474904e-05
      60145.71c -4.93629487588e-05
      95241.71c -4.89022752378e-08
      95243.71c -2.80662251846e-10
      94239.71c -0.000161857834752
      94238.71c -4.18099018163e-08
      63154.71c -9.16653261454e-08
      44103.71c -1.23001293813e-06
      92238.71c -0.0625634100325
      54135.71c -2.41199878617e-08
      62149.71c -5.83161725756e-07
      54131.71c -3.22903443003e-05
      1001.71c -0.0171288556053
      62147.71c -9.70751244319e-06
      92234.71c -5.26478800743e-08
      92235.71c -0.0132838257121
      92236.71c -0.000381751307355
      43099.71c -5.23910555062e-05
      42095.71c -4.78800009448e-05
      61147.71c -1.77770168846e-05
      40090.71c -0.90599561497
      45103.71c -2.58722500827e-05
      55133.71c -7.69322084358e-05
      93237.71c -1.94349861172e-06
      55135.71c -5.83882420377e-05
      94240.71c -1.37164238314e-05
      94241.71c -1.13705574486e-06
      94242.71c -3.92819258467e-08
m6182 62152.71c -9.03746639079e-06
      62151.71c -2.14095511512e-06
      62150.71c -1.70344838817e-05
      64155.71c -1.03272347607e-08
      63153.71c -3.195731888e-06
      60143.71c -8.60855090517e-05
      60145.71c -6.10780548833e-05
      95241.71c -1.18049088145e-07
      95243.71c -1.27519420087e-09
      94239.71c -0.00021942878913
      94238.71c -9.7746225379e-08
      63154.71c -1.50128377165e-07
      44103.71c -1.52820263098e-06
      92238.71c -0.0624951266859
      54135.71c -2.77662706225e-08
      62149.71c -5.78623525834e-07
      54131.71c -3.98942690357e-05
      1001.71c -0.0171355679083
      62147.71c -1.19397374382e-05
      92234.71c -7.22954787069e-08
      92235.71c -0.0127166755659
      92236.71c -0.000477821474125
      43099.71c -6.49357633726e-05
      42095.71c -5.94430801513e-05
      61147.71c -2.17087257194e-05
      40090.71c -0.906350648443
      45103.71c -3.20718960375e-05
      55133.71c -9.53165878042e-05
      93237.71c -3.41972898502e-06
      55135.71c -6.85205418707e-05
      94240.71c -2.34752938553e-05
      94241.71c -2.72696722958e-06
      94242.71c -1.21926556077e-07
m6183 62152.71c -1.09699634818e-05
      62151.71c -2.16608209815e-06
      62150.71c -2.0255151244e-05
      64155.71c -9.67753393417e-09
      63153.71c -3.92814912456e-06
      60143.71c -0.000100477505423
      60145.71c -7.18753035399e-05
      95241.71c -1.88257113951e-07
      95243.71c -2.96295625549e-09
      94239.71c -0.000247980564696
      94238.71c -1.60078688274e-07
      63154.71c -2.09500529647e-07
      44103.71c -1.78966810092e-06
      92238.71c -0.0624589053109
      54135.71c -3.03307198373e-08
      62149.71c -5.64174461652e-07
      54131.71c -4.68774014123e-05
      1001.71c -0.0171418045125
      62147.71c -1.40156228681e-05
      92234.71c -8.3689717191e-08
      92235.71c -0.0121966019554
      92236.71c -0.000561158273135
      43099.71c -7.65170054166e-05
      42095.71c -7.01789694505e-05
      61147.71c -2.52887769329e-05
      40090.71c -0.906680520803
      45103.71c -3.773055802e-05
      55133.71c -0.000112282157952
      93237.71c -4.68961704887e-06
      55135.71c -7.65100932034e-05
      94240.71c -3.16649460505e-05
      94241.71c -4.32672146785e-06
      94242.71c -2.36215647984e-07
m6184 62152.71c -1.1627769103e-05
      62151.71c -2.16870860665e-06
      62150.71c -2.13533357088e-05
      64155.71c -9.49070138754e-09
      63153.71c -4.18785119525e-06
      60143.71c -0.000105283647002
      60145.71c -7.55304788681e-05
      95241.71c -2.16029680554e-07
      95243.71c -3.81009178823e-09
      94239.71c -0.000256728817175
      94238.71c -1.85991151557e-07
      63154.71c -2.31813199574e-07
      44103.71c -1.87688642463e-06
      92238.71c -0.0624465921338
      54135.71c -3.1072875736e-08
      62149.71c -5.59163458101e-07
      54131.71c -4.92353212098e-05
      1001.71c -0.0171439338778
      62147.71c -1.47173020201e-05
      92234.71c -8.74594713465e-08
      92235.71c -0.0120206497608
      92236.71c -0.000589197166593
      43099.71c -8.04444322456e-05
      42095.71c -7.38310915114e-05
      61147.71c -2.6481483747e-05
      40090.71c -0.906793149207
      45103.71c -3.96433710486e-05
      55133.71c -0.00011803230296
      93237.71c -5.16110345104e-06
      55135.71c -7.89947219403e-05
      94240.71c -3.46097380422e-05
      94241.71c -4.95667590158e-06
      94242.71c -2.87985247389e-07
c **  F-19  Fuel ID= 3113
m6191 62152.71c -3.20040880874e-06
      62151.71c -1.53412948578e-06
      62150.71c -6.76653019117e-06
      64155.71c -6.9311246392e-09
      63153.71c -1.2157026302e-06
      60143.71c -3.66095696594e-05
      60145.71c -2.58902774086e-05
      95241.71c -2.26369335787e-08
      95243.71c -2.15599284603e-10
      94239.71c -8.92308094418e-05
      94238.71c -1.64451828432e-08
      63154.71c -3.40122577645e-08
      44103.71c -1.17980048792e-06
      92238.71c -0.0626178124602
      54135.71c -2.39768070462e-08
      62149.71c -6.19126056117e-07
      54131.71c -1.68281200243e-05
      1001.71c -0.0171159282109
      62147.71c -3.60555391304e-06
      92234.71c -2.7747907236e-08
      92235.71c -0.0144164351015
      92236.71c -0.000200680460379
      43099.71c -2.73649026039e-05
      42095.71c -2.24179328926e-05
      61147.71c -1.07704396802e-05
      40090.71c -0.905311847012
      45103.71c -1.30433094274e-05
      55133.71c -4.0071742832e-05
      93237.71c -7.55169389534e-07
      55135.71c -3.06570757884e-05
      94240.71c -4.98255902232e-06
      94241.71c -4.02312017605e-07
      94242.71c -1.93170766124e-08
m6192 62152.71c -4.139531096e-06
      62151.71c -1.71378935291e-06
      62150.71c -8.54696984533e-06
      64155.71c -6.92447890877e-09
      63153.71c -1.56752956215e-06
      60143.71c -4.50206698959e-05
      60145.71c -3.20071533427e-05
      95241.71c -5.11622271435e-08
      95243.71c -9.04465799801e-10
      94239.71c -0.000123742623606
      94238.71c -3.76776412444e-08
      63154.71c -5.5114575363e-08
      44103.71c -1.46951378269e-06
      92238.71c -0.0625830948969
      54135.71c -2.80369445639e-08
      62149.71c -6.22343631838e-07
      54131.71c -2.07834581038e-05
      1001.71c -0.0171194003319
      62147.71c -4.41165025984e-06
      92234.71c -3.84263637451e-08
      92235.71c -0.0141190234622
      92236.71c -0.000251405339947
      43099.71c -3.38685540956e-05
      42095.71c -2.77597282276e-05
      61147.71c -1.32310157045e-05
      40090.71c -0.905495497717
      45103.71c -1.61444808402e-05
      55133.71c -4.95816727899e-05
      93237.71c -1.30159746999e-06
      55135.71c -3.60008785571e-05
      94240.71c -8.46799174215e-06
      94241.71c -9.22826659215e-07
      94242.71c -5.60264076533e-08
m6193 62152.71c -5.08194887493e-06
      62151.71c -1.83766710297e-06
      62150.71c -1.02606012544e-05
      64155.71c -6.78462799983e-09
      63153.71c -1.90741337413e-06
      60143.71c -5.29536189045e-05
      60145.71c -3.78506135723e-05
      95241.71c -7.81942340677e-08
      95243.71c -2.00552508274e-09
      94239.71c -0.000143070924597
      94238.71c -6.11896762955e-08
      63154.71c -7.67937590247e-08
      44103.71c -1.74138497326e-06
      92238.71c -0.0625648226968
      54135.71c -3.128787939e-08
      62149.71c -6.15139840887e-07
      54131.71c -2.45524549061e-05
      1001.71c -0.0171227362983
      62147.71c -5.18064740287e-06
      92234.71c -4.49761358502e-08
      92235.71c -0.0138367182182
      92236.71c -0.000296953671562
      43099.71c -4.00872504938e-05
      42095.71c -3.28849995074e-05
      61147.71c -1.55775216993e-05
      40090.71c -0.905671946801
      45103.71c -1.90850831419e-05
      55133.71c -5.86713063057e-05
      93237.71c -1.77182149032e-06
      55135.71c -4.04167247533e-05
      94240.71c -1.1439510159e-05
      94241.71c -1.43010129225e-06
      94242.71c -1.04348351424e-07
m6194 62152.71c -5.40858858806e-06
      62151.71c -1.87276334572e-06
      62150.71c -1.08462252329e-05
      64155.71c -6.72445937682e-09
      63153.71c -2.02711715981e-06
      60143.71c -5.56274251533e-05
      60145.71c -3.9837584626e-05
      95241.71c -8.84295817765e-08
      95243.71c -2.54530143402e-09
      94239.71c -0.000149236232555
      94238.71c -7.09305396624e-08
      63154.71c -8.49374557356e-08
      44103.71c -1.83529136627e-06
      92238.71c -0.0625587271419
      54135.71c -3.23153746809e-08
      62149.71c -6.12551312283e-07
      54131.71c -2.5832109134e-05
      1001.71c -0.0171238763637
      62147.71c -5.43961252022e-06
      92234.71c -4.71206503158e-08
      92235.71c -0.013740782142
      92236.71c -0.000312382579154
      43099.71c -4.22038440603e-05
      42095.71c -3.46287318968e-05
      61147.71c -1.63700940745e-05
      40090.71c -0.905732248219
      45103.71c -2.00827771928e-05
      55133.71c -6.17644432753e-05
      93237.71c -1.94868811522e-06
      55135.71c -4.1804258175e-05
      94240.71c -1.25205979857e-05
      94241.71c -1.62603726062e-06
      94242.71c -1.25577483774e-07
c **  F-20  Fuel ID= 2949
m6201 62152.71c -6.30443194153e-07
      62151.71c -6.06660762403e-07
      62150.71c -1.33419113462e-06
      64155.71c -1.52122482201e-09
      63153.71c -2.91521098303e-07
      60143.71c -8.92535425551e-06
      60145.71c -6.79243527821e-06
      95241.71c -1.11683027676e-10
      95243.71c -1.54146532828e-13
      94239.71c -2.55601566005e-05
      94238.71c -3.98022741221e-10
      63154.71c -2.63074738511e-09
      44103.71c -1.04395942223e-06
      92238.71c -0.0626590760123
      54135.71c -2.40124222311e-08
      62149.71c -5.70596509346e-07
      54131.71c -4.21999996595e-06
      1001.71c -0.0171054606483
      62147.71c -3.05223978994e-07
      92234.71c -7.51844790617e-09
      92235.71c -0.0153389807495
      92236.71c -5.30280082106e-05
      43099.71c -7.04846369119e-06
      42095.71c -3.23186275013e-06
      61147.71c -3.27536215777e-06
      40090.71c -0.904758186803
      45103.71c -2.69694453956e-06
      55133.71c -1.01321988345e-05
      93237.71c -7.91934949955e-08
      55135.71c -8.0808199275e-06
      94240.71c -3.9737071709e-07
      94241.71c -8.75023399541e-09
      94242.71c -7.718794628e-11
m6202 62152.71c -8.27731019461e-07
      62151.71c -7.28983938402e-07
      62150.71c -1.7918452377e-06
      64155.71c -1.67479610964e-09
      63153.71c -3.7014544331e-07
      60143.71c -1.12017994497e-05
      60145.71c -8.53709347715e-06
      95241.71c -2.90464524929e-10
      95243.71c -7.51265557003e-13
      94239.71c -3.6692521879e-05
      94238.71c -8.74507964271e-10
      63154.71c -4.37480200189e-09
      44103.71c -1.31566181298e-06
      92238.71c -0.0626500352804
      54135.71c -2.85834104176e-08
      62149.71c -6.06265495996e-07
      54131.71c -5.30368567791e-06
      1001.71c -0.0171064541194
      62147.71c -3.82543497736e-07
      92234.71c -1.06127543153e-08
      92235.71c -0.0152536075191
      92236.71c -6.76899191878e-05
      43099.71c -8.86232276421e-06
      42095.71c -4.06397453599e-06
      61147.71c -4.10654773934e-06
      40090.71c -0.904810734415
      45103.71c -3.39272411581e-06
      55133.71c -1.27384992422e-05
      93237.71c -1.31316985851e-07
      55135.71c -9.62894468341e-06
      94240.71c -7.2671694833e-07
      94241.71c -2.27556821284e-08
      94242.71c -2.55590442781e-10
m6203 62152.71c -1.02774647719e-06
      62151.71c -8.33878028658e-07
      62150.71c -2.24649633544e-06
      64155.71c -1.77361848658e-09
      63153.71c -4.45493377827e-07
      60143.71c -1.33699245995e-05
      60145.71c -1.02030415426e-05
      95241.71c -4.93494029554e-10
      95243.71c -1.84572000336e-12
      94239.71c -4.36525950461e-05
      94238.71c -1.38923563853e-09
      63154.71c -6.2333962844e-09
      44103.71c -1.57406364572e-06
      92238.71c -0.0626454351837
      54135.71c -3.23784311569e-08
      62149.71c -6.22884838018e-07
      54131.71c -6.3375636054e-06
      1001.71c -0.017107401716
      62147.71c -4.56373528328e-07
      92234.71c -1.25762570173e-08
      92235.71c -0.0151727802336
      92236.71c -8.0894253597e-05
      43099.71c -1.0594821439e-05
      42095.71c -4.86019258651e-06
      61147.71c -4.89922212212e-06
      40090.71c -0.904860855589
      45103.71c -4.05577997405e-06
      55133.71c -1.52277362526e-05
      93237.71c -1.75102911665e-07
      55135.71c -1.09242261741e-05
      94240.71c -1.03185228125e-06
      94241.71c -3.86601857997e-08
      94242.71c -5.2274022359e-10
m6204 62152.71c -1.09985906779e-06
      62151.71c -8.68130708739e-07
      62150.71c -2.40718390692e-06
      64155.71c -1.80031426821e-09
      63153.71c -4.72021051889e-07
      60143.71c -1.4124915252e-05
      60145.71c -1.07839796512e-05
      95241.71c -5.78381876709e-10
      95243.71c -2.42335649852e-12
      94239.71c -4.59524999075e-05
      94238.71c -1.60934992845e-09
      63154.71c -6.9547591356e-09
      44103.71c -1.66401302812e-06
      92238.71c -0.0626439301908
      54135.71c -3.35991545242e-08
      62149.71c -6.26832439479e-07
      54131.71c -6.69807243233e-06
      1001.71c -0.0171077331964
      62147.71c -4.82077446133e-07
      92234.71c -1.32783104338e-08
      92235.71c -0.0151445961897
      92236.71c -8.54879245314e-05
      43099.71c -1.11992612918e-05
      42095.71c -5.13868788707e-06
      61147.71c -5.17493198193e-06
      40090.71c -0.904878388561
      45103.71c -4.28722951626e-06
      55133.71c -1.60961095524e-05
      93237.71c -1.92142432577e-07
      55135.71c -1.13443441616e-05
      94240.71c -1.14785086543e-06
      94241.71c -4.5322312103e-08
      94242.71c -6.49551204507e-10
c **  F-21  Fuel ID= 5649
m6211 62152.71c -5.80840461105e-06
      62151.71c -1.97740797764e-06
      62150.71c -1.15683285851e-05
      64155.71c -1.03851309197e-08
      63153.71c -2.03603640967e-06
      60143.71c -6.05185567858e-05
      60145.71c -4.24907578237e-05
      95241.71c -2.75894403904e-08
      95243.71c -1.30119573778e-10
      94239.71c -0.000143097836589
      94238.71c -2.6904458081e-08
      63154.71c -6.85645757712e-08
      44103.71c -1.21638844439e-06
      92238.71c -0.0625798588261
      54135.71c -2.41190942771e-08
      62149.71c -5.94266210236e-07
      54131.71c -2.77798532211e-05
      1001.71c -0.0171250320614
      62147.71c -7.4858430054e-06
      92234.71c -4.55869465714e-08
      92235.71c -0.013615143225
      92236.71c -0.000329036152303
      43099.71c -4.50434472888e-05
      42095.71c -4.0395493633e-05
      61147.71c -1.62049813547e-05
      40090.71c -0.905793376474
      45103.71c -2.21277665408e-05
      55133.71c -6.61167767522e-05
      93237.71c -1.46642788237e-06
      55135.71c -5.03336921746e-05
      94240.71c -1.03239278812e-05
      94241.71c -7.42285031172e-07
      94242.71c -2.1502797146e-08
m6212 62152.71c -7.55871671323e-06
      62151.71c -2.10017361127e-06
      62150.71c -1.45809190477e-05
      64155.71c -9.72793128466e-09
      63153.71c -2.66634996102e-06
      60143.71c -7.45852423476e-05
      60145.71c -5.27187896936e-05
      95241.71c -6.7643209641e-08
      95243.71c -5.99353687859e-10
      94239.71c -0.000195857186462
      94238.71c -6.26952650204e-08
      63154.71c -1.12495484801e-07
      44103.71c -1.51634675461e-06
      92238.71c -0.0625215283655
      54135.71c -2.79396928874e-08
      62149.71c -5.91985729308e-07
      54131.71c -3.44264942406e-05
      1001.71c -0.0171308585368
      62147.71c -9.23943161222e-06
      92234.71c -6.28426916652e-08
      92235.71c -0.0131193064376
      92236.71c -0.000413236997445
      43099.71c -5.59693031195e-05
      42095.71c -5.02716201421e-05
      61147.71c -1.9880443588e-05
      40090.71c -0.906101555916
      45103.71c -2.7506212978e-05
      55133.71c -8.21280086061e-05
      93237.71c -2.58182625692e-06
      55135.71c -5.92287696414e-05
      94240.71c -1.78851847595e-05
      94241.71c -1.80915707781e-06
      94242.71c -6.76404296429e-08
m6213 62152.71c -9.24044721044e-06
      62151.71c -2.14811271484e-06
      62150.71c -1.73961308658e-05
      64155.71c -9.12004595311e-09
      63153.71c -3.26846826636e-06
      60143.71c -8.7411179829e-05
      60145.71c -6.22091613697e-05
      95241.71c -1.09275903007e-07
      95243.71c -1.40938207064e-09
      94239.71c -0.000223281737519
      94238.71c -1.02584752826e-07
      63154.71c -1.57503262427e-07
      44103.71c -1.78296967602e-06
      92238.71c -0.0624905728441
      54135.71c -3.07329319496e-08
      62149.71c -5.79615052897e-07
      54131.71c -4.05759019571e-05
      1001.71c -0.0171363038614
      62147.71c -1.08789054503e-05
      92234.71c -7.29764741678e-08
      92235.71c -0.0126619130527
      92236.71c -0.000486776423813
      43099.71c -6.61212894861e-05
      42095.71c -5.94990645536e-05
      61147.71c -2.32588391472e-05
      40090.71c -0.906389575168
      45103.71c -3.24582700278e-05
      55133.71c -9.70000419773e-05
      93237.71c -3.54670143339e-06
      55135.71c -6.63233269747e-05
      94240.71c -2.43531103415e-05
      94241.71c -2.90931329168e-06
      94242.71c -1.32460413312e-07
m6214 62152.71c -9.81725358271e-06
      62151.71c -2.15730191173e-06
      62150.71c -1.83591347599e-05
      64155.71c -8.93107829782e-09
      63153.71c -3.48233079665e-06
      60143.71c -9.1721719954e-05
      60145.71c -6.54348816706e-05
      95241.71c -1.25934881364e-07
      95243.71c -1.81884138174e-09
      94239.71c -0.000231832903662
      94238.71c -1.19208948038e-07
      63154.71c -1.7449344788e-07
      44103.71c -1.87350480831e-06
      92238.71c -0.0624801694669
      54135.71c -3.15728904406e-08
      62149.71c -5.75307140676e-07
      54131.71c -4.26610057631e-05
      1001.71c -0.0171381671094
      62147.71c -1.14357063753e-05
      92234.71c -7.63820047291e-08
      92235.71c -0.0125065371313
      92236.71c -0.000511634605433
      43099.71c -6.95772450489e-05
      42095.71c -6.26481310965e-05
      61147.71c -2.43919285276e-05
      40090.71c -0.906488127843
      45103.71c -3.41393505272e-05
      55133.71c -0.000102059521188
      93237.71c -3.90481825603e-06
      55135.71c -6.85443662745e-05
      94240.71c -2.66981107332e-05
      94241.71c -3.34892613202e-06
      94242.71c -1.6205357448e-07
c **  F-22  Fuel ID= 2917
m6221 62152.71c -5.88729686085e-07
      62151.71c -5.80968732449e-07
      62150.71c -1.23110674291e-06
      64155.71c -1.3788904477e-09
      63153.71c -2.75046229485e-07
      60143.71c -8.371642985e-06
      60145.71c -6.42495885387e-06
      95241.71c -9.17733874743e-11
      95243.71c -1.26248203318e-13
      94239.71c -2.39217419368e-05
      94238.71c -3.38715119889e-10
      63154.71c -2.32342279097e-09
      44103.71c -1.04843691191e-06
      92238.71c -0.0626602141218
      54135.71c -2.41930130461e-08
      62149.71c -5.68169435091e-07
      54131.71c -3.97370437744e-06
      1001.71c -0.0171052605874
      62147.71c -2.67222683001e-07
      92234.71c -7.0392229563e-09
      92235.71c -0.0153568097644
      92236.71c -5.01049555633e-05
      43099.71c -6.656852547e-06
      42095.71c -2.86817732402e-06
      61147.71c -3.1000126854e-06
      40090.71c -0.904747604989
      45103.71c -2.49016303623e-06
      55133.71c -9.552055689e-06
      93237.71c -7.11481416626e-08
      55135.71c -7.6246888101e-06
      94240.71c -3.48016966821e-07
      94241.71c -7.309282756e-09
      94242.71c -6.39328242268e-11
m6222 62152.71c -7.6318338913e-07
      62151.71c -6.94847612788e-07
      62150.71c -1.63990794046e-06
      64155.71c -1.52219485145e-09
      63153.71c -3.45622017102e-07
      60143.71c -1.04112173625e-05
      60145.71c -7.99952918975e-06
      95241.71c -2.34534276515e-10
      95243.71c -5.92803262879e-13
      94239.71c -3.42076510769e-05
      94238.71c -7.28376054265e-10
      63154.71c -3.79758995129e-09
      44103.71c -1.30799005718e-06
      92238.71c -0.0626519331739
      54135.71c -2.85776824678e-08
      62149.71c -6.03902230781e-07
      54131.71c -4.94780195626e-06
      1001.71c -0.0171061571564
      62147.71c -3.31904219935e-07
      92234.71c -9.86439579055e-09
      92235.71c -0.0152797072187
      92236.71c -6.33762600269e-05
      43099.71c -8.29140656957e-06
      42095.71c -3.57370791555e-06
      61147.71c -3.85147922865e-06
      40090.71c -0.904795027165
      45103.71c -3.10381233013e-06
      55133.71c -1.18968264494e-05
      93237.71c -1.16074645747e-07
      55135.71c -9.02063639492e-06
      94240.71c -6.27902783556e-07
      94241.71c -1.86908883181e-08
      94242.71c -2.06066617751e-10
m6223 62152.71c -9.44110570675e-07
      62151.71c -7.95834744095e-07
      62150.71c -2.05766853208e-06
      64155.71c -1.62092625473e-09
      63153.71c -4.14847056379e-07
      60143.71c -1.24031034159e-05
      60145.71c -9.54100605271e-06
      95241.71c -3.96039738626e-10
      95243.71c -1.44916841185e-12
      94239.71c -4.06243992975e-05
      94238.71c -1.1490199431e-09
      63154.71c -5.3887195117e-09
      44103.71c -1.56121948491e-06
      92238.71c -0.0626476414529
      54135.71c -3.23188465067e-08
      62149.71c -6.21416789187e-07
      54131.71c -5.90060214871e-06
      1001.71c -0.0171070374336
      62147.71c -3.95189015859e-07
      92234.71c -1.16725584131e-08
      92235.71c -0.0152049031608
      92236.71c -7.55844729258e-05
      43099.71c -9.89193759857e-06
      42095.71c -4.26528499342e-06
      61147.71c -4.58632029513e-06
      40090.71c -0.904841587616
      45103.71c -3.70307324452e-06
      55133.71c -1.41926458491e-05
      93237.71c -1.53873905882e-07
      55135.71c -1.0220626715e-05
      94240.71c -8.88166052198e-07
      94241.71c -3.15721996895e-08
      94242.71c -4.18011477195e-10
m6224 62152.71c -1.00838089024e-06
      62151.71c -8.28706911626e-07
      62150.71c -2.20360921536e-06
      64155.71c -1.64820857532e-09
      63153.71c -4.38906882852e-07
      60143.71c -1.30870942331e-05
      60145.71c -1.00718955362e-05
      95241.71c -4.62858022518e-10
      95243.71c -1.89704625951e-12
      94239.71c -4.27336034053e-05
      94238.71c -1.32313975894e-09
      63154.71c -5.99870857909e-09
      44103.71c -1.64890300969e-06
      92238.71c -0.0626462010934
      54135.71c -3.35284207022e-08
      62149.71c -6.25566241044e-07
      54131.71c -6.22831740295e-06
      1001.71c -0.0171073424941
      62147.71c -4.1689365194e-07
      92234.71c -1.23296394616e-08
      92235.71c -0.0151791373282
      92236.71c -7.9774495347e-05
      43099.71c -1.04432468804e-05
      42095.71c -4.50320665318e-06
      61147.71c -4.83866859257e-06
      40090.71c -0.904857723165
      45103.71c -3.90912869778e-06
      55133.71c -1.49831982594e-05
      93237.71c -1.67983190531e-07
      55135.71c -1.06052072502e-05
      94240.71c -9.86190806169e-07
      94241.71c -3.69059536288e-08
      94242.71c -5.1715135624e-10
c **  F-23  Fuel ID= 5000
m6231 62152.71c -6.97478138181e-06
      62151.71c -2.04327146619e-06
      62150.71c -1.35256759956e-05
      64155.71c -1.10808677714e-08
      63153.71c -2.41760390735e-06
      60143.71c -6.99439444315e-05
      60145.71c -4.92287091601e-05
      95241.71c -4.83551153757e-08
      95243.71c -2.76226393766e-10
      94239.71c -0.000161508199012
      94238.71c -4.14504756315e-08
      63154.71c -9.11683944096e-08
      44103.71c -1.22976100741e-06
      92238.71c -0.0625637359526
      54135.71c -2.41236142171e-08
      62149.71c -5.833744525e-07
      54131.71c -3.22023279004e-05
      1001.71c -0.0171287806638
      62147.71c -9.66160163597e-06
      92234.71c -5.25102030327e-08
      92235.71c -0.0132902951365
      92236.71c -0.000380723357735
      43099.71c -5.22473402872e-05
      42095.71c -4.77336836754e-05
      61147.71c -1.77490958251e-05
      40090.71c -0.905991651092
      45103.71c -2.57992019538e-05
      55133.71c -7.67206844068e-05
      93237.71c -1.93320257041e-06
      55135.71c -5.82313711055e-05
      94240.71c -1.36443442274e-05
      94241.71c -1.12784530244e-06
      94242.71c -3.88125770171e-08
m6232 62152.71c -9.00848440691e-06
      62151.71c -2.14048976815e-06
      62150.71c -1.69866979511e-05
      64155.71c -1.0317036335e-08
      63153.71c -3.18497372125e-06
      60143.71c -8.586417764e-05
      60145.71c -6.09155291944e-05
      95241.71c -1.16772753504e-07
      95243.71c -1.25557593554e-09
      94239.71c -0.000218995465992
      94238.71c -9.69009389223e-08
      63154.71c -1.49321038977e-07
      44103.71c -1.52800250529e-06
      92238.71c -0.0624956557564
      54135.71c -2.77739616248e-08
      62149.71c -5.78870402645e-07
      54131.71c -3.9788137655e-05
      1001.71c -0.0171354758104
      62147.71c -1.1884175873e-05
      92234.71c -7.21133297672e-08
      92235.71c -0.0127245006218
      92236.71c -0.000476569162631
      43099.71c -6.47611443211e-05
      42095.71c -5.92644933279e-05
      61147.71c -2.16767118358e-05
      40090.71c -0.906345777116
      45103.71c -3.1983268424e-05
      55133.71c -9.50597783412e-05
      93237.71c -3.40170799461e-06
      55135.71c -6.8340417946e-05
      94240.71c -2.3358156033e-05
      94241.71c -2.70588420823e-06
      94242.71c -1.20510033418e-07
m6233 62152.71c -1.09364009658e-05
      62151.71c -2.1659910538e-06
      62150.71c -2.01997436708e-05
      64155.71c -9.66712896335e-09
      63153.71c -3.91476146995e-06
      60143.71c -0.000100228069357
      60145.71c -7.1688383575e-05
      95241.71c -1.86280574566e-07
      95243.71c -2.91822991575e-09
      94239.71c -0.000247533507758
      94238.71c -1.58694623826e-07
      63154.71c -2.083894195e-07
      44103.71c -1.78962148945e-06
      92238.71c -0.0624595359606
      54135.71c -3.0342439157e-08
      62149.71c -5.64468140391e-07
      54131.71c -4.67558319862e-05
      1001.71c -0.0171416975126
      62147.71c -1.39513272609e-05
      92234.71c -8.34832259505e-08
      92235.71c -0.0122055963617
      92236.71c -0.000559724136959
      43099.71c -7.63155985893e-05
      42095.71c -6.99718598025e-05
      61147.71c -2.52538218347e-05
      40090.71c -0.906674861264
      45103.71c -3.76288042015e-05
      55133.71c -0.000111986167466
      93237.71c -4.66524209849e-06
      55135.71c -7.6313668372e-05
      94240.71c -3.15135419754e-05
      94241.71c -4.29464015131e-06
      94242.71c -2.33536899136e-07
m6234 62152.71c -1.15928016474e-05
      62151.71c -2.16872374408e-06
      62150.71c -2.12954762408e-05
      64155.71c -9.47978454437e-09
      63153.71c -4.1735459487e-06
      60143.71c -0.00010502578349
      60145.71c -7.53358654825e-05
      95241.71c -2.13784081557e-07
      95243.71c -3.75304142001e-09
      94239.71c -0.000256284471773
      94238.71c -1.84387582996e-07
      63154.71c -2.30591800353e-07
      44103.71c -1.87697167768e-06
      92238.71c -0.0624472511124
      54135.71c -3.10868388105e-08
      62149.71c -5.59471881414e-07
      54131.71c -4.91089175919e-05
      1001.71c -0.0171438221049
      62147.71c -1.46501304576e-05
      92234.71c -8.72453368746e-08
      92235.71c -0.0120300141492
      92236.71c -0.000587706536729
      43099.71c -8.02345259031e-05
      42095.71c -7.36146045542e-05
      61147.71c -2.64458330043e-05
      40090.71c -0.906787237211
      45103.71c -3.95374388354e-05
      55133.71c -0.000117723875277
      93237.71c -5.13432668689e-06
      55135.71c -7.8793739388e-05
      94240.71c -3.44468431819e-05
      94241.71c -4.92046080307e-06
      94242.71c -2.84749863392e-07
c **  F-24  Fuel ID= 5948
m6241 62152.71c -7.2823128299e-06
      62151.71c -2.05534070718e-06
      62150.71c -1.40348429936e-05
      64155.71c -1.12282862414e-08
      63153.71c -2.5204702631e-06
      60143.71c -7.2373975398e-05
      60145.71c -5.09796540121e-05
      95241.71c -5.54919569114e-08
      95243.71c -3.34085199541e-10
      94239.71c -0.000166068648419
      94238.71c -4.61386260286e-08
      63154.71c -9.76500888796e-08
      44103.71c -1.23304704825e-06
      92238.71c -0.0625594848365
      54135.71c -2.40763141033e-08
      62149.71c -5.80599766047e-07
      54131.71c -3.33503636799e-05
      1001.71c -0.0171297581582
      62147.71c -1.02604360509e-05
      92234.71c -5.4305983932e-08
      92235.71c -0.0132059116495
      92236.71c -0.000394131346973
      43099.71c -5.41218797318e-05
      42095.71c -4.96421627834e-05
      61147.71c -1.81132822194e-05
      40090.71c -0.906043353655
      45103.71c -2.67520001327e-05
      55133.71c -7.94796833397e-05
      93237.71c -2.06749826777e-06
      55135.71c -6.02775061786e-05
      94240.71c -1.45845095187e-05
      94241.71c -1.24798106629e-06
      94242.71c -4.49344957724e-08
m6242 62152.71c -9.38650889415e-06
      62151.71c -2.14655948886e-06
      62150.71c -1.76099904074e-05
      64155.71c -1.04500587899e-08
      63153.71c -3.32529711958e-06
      60143.71c -8.87510984872e-05
      60145.71c -6.30354216919e-05
      95241.71c -1.33420535461e-07
      95243.71c -1.51146541241e-09
      94239.71c -0.000224647486148
      94238.71c -1.07926373921e-07
      63154.71c -1.59851498382e-07
      44103.71c -1.53061283076e-06
      92238.71c -0.0624887548624
      54135.71c -2.76736445722e-08
      62149.71c -5.75650282159e-07
      54131.71c -4.11724549241e-05
      1001.71c -0.0171366770824
      62147.71c -1.26088892776e-05
      92234.71c -7.4489176771e-08
      92235.71c -0.0126224350502
      92236.71c -0.000492903600344
      43099.71c -6.70387757514e-05
      42095.71c -6.1593878116e-05
      61147.71c -2.20942826962e-05
      40090.71c -0.906409315929
      45103.71c -3.31392765247e-05
      55133.71c -9.84094546769e-05
      93237.71c -3.6367635273e-06
      55135.71c -7.06898518064e-05
      94240.71c -2.48860350561e-05
      94241.71c -2.98087912811e-06
      94242.71c -1.38986348017e-07
m6243 62152.71c -1.13741713037e-05
      62151.71c -2.1671785844e-06
      62150.71c -2.09224484902e-05
      64155.71c -9.80284547519e-09
      63153.71c -4.08938241001e-06
      60143.71c -0.000103481571298
      60145.71c -7.41264611122e-05
      95241.71c -2.12061428309e-07
      95243.71c -3.50161481167e-09
      94239.71c -0.00025336466377
      94238.71c -1.76747572011e-07
      63154.71c -2.22882107286e-07
      44103.71c -1.79022946288e-06
      92238.71c -0.0624513101254
      54135.71c -3.01895790269e-08
      62149.71c -5.60637562221e-07
      54131.71c -4.83415143246e-05
      1001.71c -0.0171430931581
      62147.71c -1.47899625338e-05
      92234.71c -8.61765800585e-08
      92235.71c -0.0120882784495
      92236.71c -0.000578430192241
      43099.71c -7.89426345077e-05
      42095.71c -7.26732800651e-05
      61147.71c -2.57097562221e-05
      40090.71c -0.906748681065
      45103.71c -3.89560230429e-05
      55133.71c -0.000115846898739
      93237.71c -4.98317506537e-06
      55135.71c -7.88757219729e-05
      94240.71c -3.34883704792e-05
      94241.71c -4.71309056821e-06
      94242.71c -2.68476973072e-07
m6244 62152.71c -1.20488972182e-05
      62151.71c -2.16852629999e-06
      62150.71c -2.20501621803e-05
      64155.71c -9.62217762755e-09
      63153.71c -4.36013543529e-06
      60143.71c -0.000108389208236
      60145.71c -7.78742916181e-05
      95241.71c -2.43074395596e-07
      95243.71c -4.49717392261e-09
      94239.71c -0.000262080260067
      94238.71c -2.05303617786e-07
      63154.71c -2.46523035985e-07
      44103.71c -1.87585968553e-06
      92238.71c -0.0624386557717
      54135.71c -3.09047124213e-08
      62149.71c -5.55448983426e-07
      54131.71c -5.07576543745e-05
      1001.71c -0.0171452800065
      62147.71c -1.55262780519e-05
      92234.71c -9.00383849376e-08
      92235.71c -0.0119078704017
      92236.71c -0.0006071494635
      43099.71c -8.29724246553e-05
      42095.71c -7.64383370953e-05
      61147.71c -2.69108409823e-05
      40090.71c -0.90686434992
      45103.71c -4.09191582768e-05
      55133.71c -0.000121746830265
      93237.71c -5.48358754427e-06
      55135.71c -8.14152413049e-05
      94240.71c -3.65715509417e-05
      94241.71c -5.39282991709e-06
      94242.71c -3.26950369123e-07
c **  F-25  Fuel ID= 5022
m6251 62152.71c -6.61743870231e-06
      62151.71c -2.02580326624e-06
      62150.71c -1.29296326624e-05
      64155.71c -1.08865669643e-08
      63153.71c -2.29978843452e-06
      60143.71c -6.70855878005e-05
      60145.71c -4.71776338628e-05
      95241.71c -4.12769130581e-08
      95243.71c -2.23666468023e-10
      94239.71c -0.000156003579303
      94238.71c -3.66246470982e-08
      63154.71c -8.40087580175e-08
      44103.71c -1.22553943793e-06
      92238.71c -0.0625686869976
      54135.71c -2.40729395707e-08
      62149.71c -5.86786508791e-07
      54131.71c -3.08573625997e-05
      1001.71c -0.0171276375068
      62147.71c -8.98309226961e-06
      92234.71c -5.04095875985e-08
      92235.71c -0.013389176121
      92236.71c -0.000365000365354
      43099.71c -5.0053960751e-05
      42095.71c -4.54990742985e-05
      61147.71c -1.72975937192e-05
      40090.71c -0.905931186156
      45103.71c -2.46822632714e-05
      55133.71c -7.34930470407e-05
      93237.71c -1.78521763813e-06
      55135.71c -5.58300631143e-05
      94240.71c -1.25975439609e-05
      94241.71c -1.00149172416e-06
      94242.71c -3.28494233853e-08
m6252 62152.71c -8.56622612934e-06
      62151.71c -2.13051202054e-06
      62150.71c -1.62551018627e-05
      64155.71c -1.01500467751e-08
      63153.71c -3.02454125883e-06
      60143.71c -8.24532771647e-05
      60145.71c -5.84243706933e-05
      95241.71c -1.00116141139e-07
      95243.71c -1.02013722685e-09
      94239.71c -0.000212133775067
      94238.71c -8.55511948416e-08
      63154.71c -1.37663025172e-07
      44103.71c -1.52388385805e-06
      92238.71c -0.0625035720543
      54135.71c -2.77572892797e-08
      62149.71c -5.83019112028e-07
      54131.71c -3.81610182334e-05
      1001.71c -0.0171340694708
      62147.71c -1.10608119253e-05
      92234.71c -6.93067611819e-08
      92235.71c -0.0128444750353
      92236.71c -0.000457341896183
      43099.71c -6.20882994672e-05
      42095.71c -5.65302481162e-05
      61147.71c -2.11543227095e-05
      40090.71c -0.906271391671
      45103.71c -3.0624153701e-05
      55133.71c -9.11299610558e-05
      93237.71c -3.14236500399e-06
      55135.71c -6.55746705862e-05
      94240.71c -2.16414398557e-05
      94241.71c -2.41393745444e-06
      94242.71c -1.02372604417e-07
m6253 62152.71c -1.04212493655e-05
      62151.71c -2.16242643342e-06
      62150.71c -1.93487803997e-05
      64155.71c -9.50554344002e-09
      63153.71c -3.71484183442e-06
      60143.71c -9.63636185387e-05
      60145.71c -6.8814013484e-05
      95241.71c -1.60290804542e-07
      95243.71c -2.37988327326e-09
      94239.71c -0.000240397185596
      94238.71c -1.40089341348e-07
      63154.71c -1.92311181362e-07
      44103.71c -1.78782776367e-06
      92238.71c -0.0624690125666
      54135.71c -3.04179286343e-08
      62149.71c -5.69264018528e-07
      54131.71c -4.48845303758e-05
      1001.71c -0.0171400580505
      62147.71c -1.29962444054e-05
      92234.71c -8.03105405718e-08
      92235.71c -0.0123439600238
      92236.71c -0.000537629777031
      43099.71c -7.32227991575e-05
      42095.71c -6.67923513067e-05
      61147.71c -2.46771476021e-05
      40090.71c -0.906588145289
      45103.71c -3.60623330529e-05
      55133.71c -0.000107441167242
      93237.71c -4.31216500354e-06
      55135.71c -7.32876904609e-05
      94240.71c -2.92781933647e-05
      94241.71c -3.84615252211e-06
      94242.71c -1.99005618803e-07
m6254 62152.71c -1.1054347482e-05
      62151.71c -2.16694551527e-06
      62150.71c -2.04050977991e-05
      64155.71c -9.31436102138e-09
      63153.71c -3.95980878273e-06
      60143.71c -0.000101019515442
      60145.71c -7.23367884581e-05
      95241.71c -1.84180767904e-07
      95243.71c -3.0639376667e-09
      94239.71c -0.000249093999509
      94238.71c -1.62787207882e-07
      63154.71c -2.12886570496e-07
      44103.71c -1.87664868807e-06
      92238.71c -0.0624574462566
      54135.71c -3.12044582176e-08
      62149.71c -5.64479438224e-07
      54131.71c -4.71584754824e-05
      1001.71c -0.0171421023803
      62147.71c -1.36514120125e-05
      92234.71c -8.39666846953e-08
      92235.71c -0.0121743548098
      92236.71c -0.000564693636924
      43099.71c -7.70043426896e-05
      42095.71c -7.02869425557e-05
      61147.71c -2.58538857445e-05
      40090.71c -0.906696275909
      45103.71c -3.7903362424e-05
      55133.71c -0.00011297753516
      93237.71c -4.74616150857e-06
      55135.71c -7.56921864466e-05
      94240.71c -3.20317870297e-05
      94241.71c -4.41297853843e-06
      94242.71c -2.42902311143e-07
c **  F-26  Fuel ID= 2907
m6261 62152.71c -7.94767650917e-07
      62151.71c -7.09619820048e-07
      62150.71c -1.74059948926e-06
      64155.71c -2.03992400718e-09
      63153.71c -3.56454574763e-07
      60143.71c -1.1085611642e-05
      60145.71c -8.25106238576e-06
      95241.71c -1.81323648674e-10
      95243.71c -2.51510958868e-13
      94239.71c -3.11317317252e-05
      94238.71c -6.08150564242e-10
      63154.71c -3.78648775174e-09
      44103.71c -1.05694108619e-06
      92238.71c -0.0626557607034
      54135.71c -2.38741377296e-08
      62149.71c -5.81711390418e-07
      54131.71c -5.18805367402e-06
      1001.71c -0.0171062552864
      62147.71c -4.45229992225e-07
      92234.71c -9.19871592158e-09
      92235.71c -0.0152683722397
      92236.71c -6.44287224163e-05
      43099.71c -8.59747987636e-06
      42095.71c -4.58044589815e-06
      61147.71c -3.97208328308e-06
      40090.71c -0.904800217547
      45103.71c -3.48645635945e-06
      55133.71c -1.2418288525e-05
      93237.71c -1.073088285e-07
      55135.71c -9.83338419342e-06
      94240.71c -5.74673551688e-07
      94241.71c -1.37831760262e-08
      94242.71c -1.24719932958e-10
m6262 62152.71c -1.05600930163e-06
      62151.71c -8.49272774967e-07
      62150.71c -2.33005308325e-06
      64155.71c -2.21430859789e-09
      63153.71c -4.56580902834e-07
      60143.71c -1.39977728168e-05
      60145.71c -1.04361149372e-05
      95241.71c -4.81543291721e-10
      95243.71c -1.26783565262e-12
      94239.71c -4.48804954033e-05
      94238.71c -1.36757163172e-09
      63154.71c -6.3750450159e-09
      44103.71c -1.34058476784e-06
      92238.71c -0.0626445227082
      54135.71c -2.85332577316e-08
      62149.71c -6.13000915508e-07
      54131.71c -6.56150348248e-06
      1001.71c -0.0171074923922
      62147.71c -5.61685279052e-07
      92234.71c -1.30358197073e-08
      92235.71c -0.0151615942563
      92236.71c -8.27342658235e-05
      43099.71c -1.08794293933e-05
      42095.71c -5.79902009611e-06
      61147.71c -5.00892658429e-06
      40090.71c -0.904865651719
      45103.71c -4.41474805442e-06
      55133.71c -1.57129654869e-05
      93237.71c -1.82287223778e-07
      55135.71c -1.17721863797e-05
      94240.71c -1.06300948071e-06
      94241.71c -3.65783198088e-08
      94242.71c -4.24657532923e-10
m6263 62152.71c -1.31643770423e-06
      62151.71c -9.63461522208e-07
      62150.71c -2.89786138316e-06
      64155.71c -2.30928520078e-09
      63153.71c -5.50749771252e-07
      60143.71c -1.67109164802e-05
      60145.71c -1.24794776023e-05
      95241.71c -8.19094370814e-10
      95243.71c -3.14204924359e-12
      94239.71c -5.32856367325e-05
      94238.71c -2.188165146e-09
      63154.71c -9.09684145784e-09
      44103.71c -1.60517970807e-06
      92238.71c -0.0626387284019
      54135.71c -3.23240809675e-08
      62149.71c -6.26101741328e-07
      54131.71c -7.84445251857e-06
      1001.71c -0.0171086526603
      62147.71c -6.70535575306e-07
      92234.71c -1.54551561889e-08
      92235.71c -0.0150624851051
      92236.71c -9.89177995896e-05
      43099.71c -1.30140695937e-05
      42095.71c -6.94063572101e-06
      61147.71c -5.97649318825e-06
      40090.71c -0.904927021716
      45103.71c -5.2805203772e-06
      55133.71c -1.87942506575e-05
      93237.71c -2.45468697722e-07
      55135.71c -1.33577561534e-05
      94240.71c -1.50906558269e-06
      94241.71c -6.21804889909e-08
      94242.71c -8.70331083834e-10
m6264 62152.71c -1.40814401651e-06
      62151.71c -9.99605895475e-07
      62150.71c -3.09347738367e-06
      64155.71c -2.33297753612e-09
      63153.71c -5.83261423039e-07
      60143.71c -1.76351625733e-05
      60145.71c -1.31764278512e-05
      95241.71c -9.60453720508e-10
      95243.71c -4.10520796792e-12
      94239.71c -5.60635790846e-05
      94238.71c -2.52314159685e-09
      63154.71c -1.01241461964e-08
      44103.71c -1.6952125658e-06
      92238.71c -0.0626367922621
      54135.71c -3.35064328902e-08
      62149.71c -6.29021892559e-07
      54131.71c -8.2821025422e-06
      1001.71c -0.0171090490717
      62147.71c -7.07632973219e-07
      92234.71c -1.63165121355e-08
      92235.71c -0.0150287289002
      92236.71c -0.000104419361166
      43099.71c -1.37427324596e-05
      42095.71c -7.33083819904e-06
      61147.71c -6.30598771385e-06
      40090.71c -0.90494798908
      45103.71c -5.57627508641e-06
      55133.71c -1.98462312109e-05
      93237.71c -2.68659260573e-07
      55135.71c -1.38604490778e-05
      94240.71c -1.6767820889e-06
      94241.71c -7.2893486986e-08
      94242.71c -1.08059821191e-09
c **  F-27  Fuel ID= 5944
m6271 62152.71c -6.94405981122e-06
      62151.71c -2.02704788635e-06
      62150.71c -1.34577468486e-05
      64155.71c -1.10898384976e-08
      63153.71c -2.41904750307e-06
      60143.71c -6.95494738365e-05
      60145.71c -4.8988035024e-05
      95241.71c -5.55993244997e-08
      95243.71c -3.65968188079e-10
      94239.71c -0.000160945856274
      94238.71c -4.5022356331e-08
      63154.71c -9.27404035151e-08
      44103.71c -1.21434644118e-06
      92238.71c -0.0625628822151
      54135.71c -2.38692872683e-08
      62149.71c -5.84278666567e-07
      54131.71c -3.20412489656e-05
      1001.71c -0.0171286594911
      62147.71c -9.7577641615e-06
      92234.71c -5.25150703019e-08
      92235.71c -0.0133017732434
      92236.71c -0.000379045875665
      43099.71c -5.19996346774e-05
      42095.71c -4.75515640857e-05
      61147.71c -1.75058688654e-05
      40090.71c -0.905985241914
      45103.71c -2.56775066833e-05
      55133.71c -7.63565154047e-05
      93237.71c -1.98209238303e-06
      55135.71c -5.80203908512e-05
      94240.71c -1.38469386853e-05
      94241.71c -1.20138622737e-06
      94242.71c -4.52548099332e-08
m6272 62152.71c -9.01371840076e-06
      62151.71c -2.12916523608e-06
      62150.71c -1.69927904146e-05
      64155.71c -1.03471239841e-08
      63153.71c -3.21184541263e-06
      60143.71c -8.57649469439e-05
      60145.71c -6.09194017243e-05
      95241.71c -1.34650781628e-07
      95243.71c -1.68072885422e-09
      94239.71c -0.000218820944528
      94238.71c -1.06712533238e-07
      63154.71c -1.53489701402e-07
      44103.71c -1.51696768171e-06
      92238.71c -0.0624936280244
      54135.71c -2.75941787811e-08
      62149.71c -5.7944124103e-07
      54131.71c -3.97809434652e-05
      1001.71c -0.017135502935
      62147.71c -1.20561465589e-05
      92234.71c -7.22058641002e-08
      92235.71c -0.0127243378878
      92236.71c -0.000476626210056
      43099.71c -6.47795692313e-05
      42095.71c -5.93359635626e-05
      61147.71c -2.1479152598e-05
      40090.71c -0.906347211817
      45103.71c -3.19889582795e-05
      55133.71c -9.5085471902e-05
      93237.71c -3.51573008977e-06
      55135.71c -6.83359449651e-05
      94240.71c -2.38400234345e-05
      94241.71c -2.89738657561e-06
      94242.71c -1.41932891319e-07
m6273 62152.71c -1.09511349156e-05
      62151.71c -2.15802653111e-06
      62150.71c -2.0235048164e-05
      64155.71c -9.7317294805e-09
      63153.71c -3.95650735198e-06
      60143.71c -0.000100200793725
      60145.71c -7.17821855906e-05
      95241.71c -2.12608057451e-07
      95243.71c -3.87498790791e-09
      94239.71c -0.000246926664387
      94238.71c -1.75650526952e-07
      63154.71c -2.14666473211e-07
      44103.71c -1.77683279071e-06
      92238.71c -0.0624575003124
      54135.71c -3.01811603235e-08
      62149.71c -5.64767389399e-07
      54131.71c -4.67992014004e-05
      1001.71c -0.017141785835
      62147.71c -1.41672977316e-05
      92234.71c -8.38102712751e-08
      92235.71c -0.0122009117886
      92236.71c -0.000560452972711
      43099.71c -7.64328433904e-05
      42095.71c -7.01562441492e-05
      61147.71c -2.50459688424e-05
      40090.71c -0.9066795329
      45103.71c -3.76766455167e-05
      55133.71c -0.000112154693919
      93237.71c -4.83866578403e-06
      55135.71c -7.6350567922e-05
      94240.71c -3.20730619393e-05
      94241.71c -4.56568427137e-06
      94242.71c -2.72832481545e-07
m6274 62152.71c -1.15971998576e-05
      62151.71c -2.16162949557e-06
      62150.71c -2.13188266179e-05
      64155.71c -9.55188590907e-09
      63153.71c -4.21580905468e-06
      60143.71c -0.000104920807607
      60145.71c -7.53861820273e-05
      95241.71c -2.43448544137e-07
      95243.71c -4.97110113812e-09
      94239.71c -0.000255228966393
      94238.71c -2.03486781093e-07
      63154.71c -2.37310426194e-07
      44103.71c -1.86355709695e-06
      92238.71c -0.0624455909543
      54135.71c -3.09443840132e-08
      62149.71c -5.59725273332e-07
      54131.71c -4.91206768325e-05
      1001.71c -0.0171438883817
      62147.71c -1.48667813776e-05
      92234.71c -8.79069454804e-08
      92235.71c -0.0120273038434
      92236.71c -0.000588143890437
      43099.71c -8.03074081155e-05
      42095.71c -7.37594483593e-05
      61147.71c -2.6211005433e-05
      40090.71c -0.906790742786
      45103.71c -3.95565531922e-05
      55133.71c -0.000117826083198
      93237.71c -5.30496876472e-06
      55135.71c -7.87929657165e-05
      94240.71c -3.49645981387e-05
      94241.71c -5.2174270095e-06
      94242.71c -3.31904866583e-07
c **  F-28  Fuel ID= 3326
m6281 62152.71c -3.75431039697e-06
      62151.71c -1.75665705201e-06
      62150.71c -7.97251779354e-06
      64155.71c -8.45355519334e-09
      63153.71c -1.38056042832e-06
      60143.71c -4.29005452783e-05
      60145.71c -3.00860123958e-05
      95241.71c -7.28558559993e-09
      95243.71c -2.24226555539e-11
      94239.71c -0.000106157282531
      94238.71c -1.00560994414e-08
      63154.71c -3.51219925753e-08
      44103.71c -1.18983718854e-06
      92238.71c -0.0626085338094
      54135.71c -2.41320928715e-08
      62149.71c -6.13662142119e-07
      54131.71c -1.96159476576e-05
      1001.71c -0.0171181822033
      62147.71c -4.02773969001e-06
      92234.71c -3.26494938848e-08
      92235.71c -0.0142135330499
      92236.71c -0.000233546723655
      43099.71c -3.18074379174e-05
      42095.71c -2.69178453182e-05
      61147.71c -1.27577564861e-05
      40090.71c -0.905431067308
      45103.71c -1.53545785683e-05
      55133.71c -4.66248339933e-05
      93237.71c -7.77781981379e-07
      55135.71c -3.57231752844e-05
      94240.71c -5.32087858572e-06
      94241.71c -2.74434391588e-07
      94242.71c -5.3895661484e-09
m6282 62152.71c -4.94725968637e-06
      62151.71c -1.92988766728e-06
      62150.71c -1.01326794665e-05
      64155.71c -8.17410390456e-09
      63153.71c -1.78590275756e-06
      60143.71c -5.32400187482e-05
      60145.71c -3.75050768892e-05
      95241.71c -1.83637948216e-08
      95243.71c -1.06234210472e-10
      94239.71c -0.000147830175714
      94238.71c -2.31573169e-08
      63154.71c -5.78391417474e-08
      44103.71c -1.48890233295e-06
      92238.71c -0.0625678581536
      54135.71c -2.82248925207e-08
      62149.71c -6.15124465606e-07
      54131.71c -2.4437808015e-05
      1001.71c -0.0171223697247
      62147.71c -5.00256039235e-06
      92234.71c -4.5397633383e-08
      92235.71c -0.0138528674821
      92236.71c -0.000295060677818
      43099.71c -3.96944207926e-05
      42095.71c -3.36419876796e-05
      61147.71c -1.57778321635e-05
      40090.71c -0.905652557647
      45103.71c -1.91772866882e-05
      55133.71c -5.81735049014e-05
      93237.71c -1.36386852172e-06
      55135.71c -4.22333198862e-05
      94240.71c -9.42132909149e-06
      94241.71c -6.88746744996e-07
      94242.71c -1.73590429565e-08
m6283 62152.71c -6.12540004237e-06
      62151.71c -2.02994342793e-06
      62150.71c -1.21748522318e-05
      64155.71c -7.79301121269e-09
      63153.71c -2.17492596722e-06
      60143.71c -6.2835971562e-05
      60145.71c -4.44680985985e-05
      95241.71c -3.03355228779e-08
      95243.71c -2.54347489721e-10
      94239.71c -0.000171210385948
      94238.71c -3.76975462691e-08
      63154.71c -8.14507517174e-08
      44103.71c -1.76245284508e-06
      92238.71c -0.0625464144898
      54135.71c -3.14037505408e-08
      62149.71c -6.06671250916e-07
      54131.71c -2.89540552905e-05
      1001.71c -0.0171263202525
      62147.71c -5.92210710407e-06
      92234.71c -5.30483825149e-08
      92235.71c -0.0135167530248
      92236.71c -0.000349420390819
      43099.71c -4.71037556342e-05
      42095.71c -3.99889169725e-05
      61147.71c -1.85967988888e-05
      40090.71c -0.905861512695
      45103.71c -2.27472754375e-05
      55133.71c -6.90198825784e-05
      93237.71c -1.87100615773e-06
      55135.71c -4.75267260869e-05
      94240.71c -1.30491957491e-05
      94241.71c -1.13411112291e-06
      94242.71c -3.46308782444e-08
m6284 62152.71c -6.53739970215e-06
      62151.71c -2.0557975807e-06
      62150.71c -1.28790997835e-05
      64155.71c -7.65735543492e-09
      63153.71c -2.31310708905e-06
      60143.71c -6.61045021596e-05
      60145.71c -4.68574910685e-05
      95241.71c -3.52215633852e-08
      95243.71c -3.30836661184e-10
      94239.71c -0.000178750546341
      94238.71c -4.37737915625e-08
      63154.71c -9.04613616035e-08
      44103.71c -1.85688218346e-06
      92238.71c -0.0625392321932
      54135.71c -3.24040423821e-08
      62149.71c -6.03603924089e-07
      54131.71c -3.05012305325e-05
      1001.71c -0.017127682178
      62147.71c -6.23738973135e-06
      92234.71c -5.56738280741e-08
      92235.71c -0.0134014816328
      92236.71c -0.000367985017054
      43099.71c -4.96488675026e-05
      42095.71c -4.21728242601e-05
      61147.71c -1.9555144117e-05
      40090.71c -0.905933548947
      45103.71c -2.39713472925e-05
      55133.71c -7.27435218903e-05
      93237.71c -2.06319032074e-06
      55135.71c -4.92064937146e-05
      94240.71c -1.43877027513e-05
      94241.71c -1.31572524204e-06
      94242.71c -4.26414377434e-08
c **  F-29  Fuel ID= 2914
m6291 62152.71c -9.35658332975e-07
      62151.71c -7.97647594785e-07
      62150.71c -2.08759169473e-06
      64155.71c -2.45991616585e-09
      63153.71c -4.11896352304e-07
      60143.71c -1.29240975906e-05
      60145.71c -9.50072496401e-06
      95241.71c -2.36663622494e-10
      95243.71c -3.27135119543e-13
      94239.71c -3.55456553064e-05
      94238.71c -7.75753661086e-10
      63154.71c -4.75073773328e-09
      44103.71c -1.07922750717e-06
      92238.71c -0.0626533813399
      54135.71c -2.39390410107e-08
      62149.71c -5.91979784658e-07
      54131.71c -6.01438981802e-06
      1001.71c -0.0171069357039
      62147.71c -5.60615059413e-07
      92234.71c -1.04930607529e-08
      92235.71c -0.0152080159336
      92236.71c -7.40821957876e-05
      43099.71c -9.92293692637e-06
      42095.71c -5.69733785849e-06
      61147.71c -4.57019930333e-06
      40090.71c -0.904836206837
      45103.71c -4.15141605643e-06
      55133.71c -1.43716248655e-05
      93237.71c -1.29800618875e-07
      55135.71c -1.13060210594e-05
      94240.71c -7.18562930209e-07
      94241.71c -1.7786645218e-08
      94242.71c -1.63561974759e-10
m6292 62152.71c -1.23491348334e-06
      62151.71c -9.44119778181e-07
      62150.71c -2.75280501475e-06
      64155.71c -2.64117549158e-09
      63153.71c -5.24182475788e-07
      60143.71c -1.61969370865e-05
      60145.71c -1.19276355955e-05
      95241.71c -6.16579466924e-10
      95243.71c -1.61315958527e-12
      94239.71c -5.08622717858e-05
      94238.71c -1.71530410624e-09
      63154.71c -7.8955296892e-09
      44103.71c -1.35880986104e-06
      92238.71c -0.0626406582298
      54135.71c -2.84391488063e-08
      62149.71c -6.18079609852e-07
      54131.71c -7.5501418537e-06
      1001.71c -0.0171083084354
      62147.71c -7.02107725956e-07
      92234.71c -1.4735451353e-08
      92235.71c -0.0150894091255
      92236.71c -9.44340611029e-05
      43099.71c -1.24639368595e-05
      42095.71c -7.16085007966e-06
      61147.71c -5.71945425485e-06
      40090.71c -0.904908814645
      45103.71c -5.21830401491e-06
      55133.71c -1.80502299309e-05
      93237.71c -2.18516501316e-07
      55135.71c -1.34574759239e-05
      94240.71c -1.31183715138e-06
      94241.71c -4.63063572554e-08
      94242.71c -5.42728784486e-10
m6293 62152.71c -1.54069817602e-06
      62151.71c -1.06457951845e-06
      62150.71c -3.4049955129e-06
      64155.71c -2.73210523538e-09
      63153.71c -6.32191730528e-07
      60143.71c -1.93168332538e-05
      60145.71c -1.42503225166e-05
      95241.71c -1.04643236486e-09
      95243.71c -3.96287225913e-12
      94239.71c -6.02614435655e-05
      94238.71c -2.75105557282e-09
      63154.71c -1.12491036438e-08
      44103.71c -1.62520409011e-06
      92238.71c -0.0626341021218
      54135.71c -3.2175096542e-08
      62149.71c -6.28000364999e-07
      54131.71c -9.01819138576e-06
      1001.71c -0.0171096243164
      62147.71c -8.37484708875e-07
      92234.71c -1.74735927493e-08
      92235.71c -0.0149768381345
      92236.71c -0.000112792870056
      43099.71c -1.48966482316e-05
      42095.71c -8.56468151993e-06
      61147.71c -6.81696382626e-06
      40090.71c -0.90497841547
      45103.71c -6.2366534217e-06
      55133.71c -2.15712788657e-05
      93237.71c -2.95696597596e-07
      55135.71c -1.52586850965e-05
      94240.71c -1.85943425215e-06
      94241.71c -7.85592228217e-08
      94242.71c -1.1102350692e-09
m6294 62152.71c -1.64930328941e-06
      62151.71c -1.10268393565e-06
      62150.71c -3.63126753913e-06
      64155.71c -2.752449466e-09
      63153.71c -6.69998680185e-07
      60143.71c -2.03874138437e-05
      60145.71c -1.50497863216e-05
      95241.71c -1.23227510402e-09
      95243.71c -5.25916804102e-12
      94239.71c -6.34539764468e-05
      94238.71c -3.18869655821e-09
      63154.71c -1.25346134033e-08
      44103.71c -1.71723033135e-06
      92238.71c -0.0626318876857
      54135.71c -3.33734529299e-08
      62149.71c -6.3005742963e-07
      54131.71c -9.52317214867e-06
      1001.71c -0.0171100774989
      62147.71c -8.83974109977e-07
      92234.71c -1.84467826996e-08
      92235.71c -0.0149381257587
      92236.71c -0.000119083633326
      43099.71c -1.57342344955e-05
      42095.71c -9.04833751835e-06
      61147.71c -7.19343096338e-06
      40090.71c -0.905002385625
      45103.71c -6.58716964335e-06
      55133.71c -2.27835233604e-05
      93237.71c -3.25423989547e-07
      55135.71c -1.58343698228e-05
      94240.71c -2.06901412975e-06
      94241.71c -9.25113459145e-08
      94242.71c -1.38504737836e-09
c **  F-30  Fuel ID= 2909
m6301 62152.71c -6.28239000735e-07
      62151.71c -6.05121114047e-07
      62150.71c -1.32865340242e-06
      64155.71c -1.50866057386e-09
      63153.71c -2.90580067028e-07
      60143.71c -8.89171687767e-06
      60145.71c -6.77220196803e-06
      95241.71c -1.09844489984e-10
      95243.71c -1.51483457268e-13
      94239.71c -2.5348250945e-05
      94238.71c -3.92704077234e-10
      63154.71c -2.60947202523e-09
      44103.71c -1.04661766108e-06
      92238.71c -0.0626592302716
      54135.71c -2.40657763775e-08
      62149.71c -5.70221387424e-07
      54131.71c -4.20567667338e-06
      1001.71c -0.0171054507598
      62147.71c -3.02412336579e-07
      92234.71c -7.47130058342e-09
      92235.71c -0.0153399847938
      92236.71c -5.28364235936e-05
      43099.71c -7.02649028296e-06
      42095.71c -3.20502205954e-06
      61147.71c -3.26560255304e-06
      40090.71c -0.904757663767
      45103.71c -2.68306160544e-06
      55133.71c -1.00989318559e-05
      93237.71c -7.83538940793e-08
      55135.71c -8.04894676888e-06
      94240.71c -3.93033892183e-07
      94241.71c -8.61591213096e-09
      94242.71c -7.61976019345e-11
m6302 62152.71c -8.20702876568e-07
      62151.71c -7.2485504976e-07
      62150.71c -1.77514325264e-06
      64155.71c -1.65841938412e-09
      63153.71c -3.67367664558e-07
      60143.71c -1.11133408309e-05
      60145.71c -8.47636874063e-06
      95241.71c -2.84116071928e-10
      95243.71c -7.29408732257e-13
      94239.71c -3.63193581742e-05
      94238.71c -8.5599956408e-10
      63154.71c -4.31008510365e-09
      44103.71c -1.31351514488e-06
      92238.71c -0.0626503367241
      54135.71c -2.85671155916e-08
      62149.71c -6.05546354659e-07
      54131.71c -5.26366387157e-06
      1001.71c -0.0171064205629
      62147.71c -3.77466736454e-07
      92234.71c -1.04874012519e-08
      92235.71c -0.0152565819579
      92236.71c -6.71768594978e-05
      43099.71c -8.79793708779e-06
      42095.71c -4.01370605987e-06
      61147.71c -4.07734986534e-06
      40090.71c -0.90480895951
      45103.71c -3.36123465421e-06
      55133.71c -1.26437670136e-05
      93237.71c -1.29308004829e-07
      55135.71c -9.56078692689e-06
      94240.71c -7.14268337111e-07
      94241.71c -2.22848468578e-08
      94242.71c -2.49954302308e-10
m6303 62152.71c -1.01871488072e-06
      62151.71c -8.29340662529e-07
      62150.71c -2.22586044455e-06
      64155.71c -1.75724228599e-09
      63153.71c -4.42085355425e-07
      60143.71c -1.32631349505e-05
      60145.71c -1.01293942506e-05
      95241.71c -4.79906048013e-10
      95243.71c -1.78752613149e-12
      94239.71c -4.31910863663e-05
      94238.71c -1.35806947743e-09
      63154.71c -6.13521496833e-09
      44103.71c -1.57141855366e-06
      92238.71c -0.0626457391011
      54135.71c -3.23558763263e-08
      62149.71c -6.22335420725e-07
      54131.71c -6.28914822266e-06
      1001.71c -0.0171073627259
      62147.71c -4.50249952712e-07
      92234.71c -1.24402502613e-08
      92235.71c -0.0151763740501
      92236.71c -8.0263013217e-05
      43099.71c -1.05167830707e-05
      42095.71c -4.79913824136e-06
      61147.71c -4.86398358841e-06
      40090.71c -0.904858793285
      45103.71c -4.01759874219e-06
      55133.71c -1.51129331984e-05
      93237.71c -1.72373903378e-07
      55135.71c -1.08461745631e-05
      94240.71c -1.01338749161e-06
      94241.71c -3.76462978015e-08
      94242.71c -5.07703475113e-10
m6304 62152.71c -1.08929384691e-06
      62151.71c -8.63342181381e-07
      62150.71c -2.38363765908e-06
      64155.71c -1.78467484252e-09
      63153.71c -4.68110278506e-07
      60143.71c -1.40047536116e-05
      60145.71c -1.07006414565e-05
      95241.71c -5.61740166825e-10
      95243.71c -2.32912541263e-12
      94239.71c -4.54170584604e-05
      94238.71c -1.56354554915e-09
      63154.71c -6.83242223921e-09
      44103.71c -1.66067764673e-06
      92238.71c -0.0626443693044
      54135.71c -3.35674831358e-08
      62149.71c -6.26336696765e-07
      54131.71c -6.64342166171e-06
      1001.71c -0.0171076872215
      62147.71c -4.75343144843e-07
      92234.71c -1.3089208946e-08
      92235.71c -0.0151486840058
      92236.71c -8.47685672462e-05
      43099.71c -1.11109719633e-05
      42095.71c -5.07055464419e-06
      61147.71c -5.1353688883e-06
      40090.71c -0.904875956815
      45103.71c -4.24434097703e-06
      55133.71c -1.59664893315e-05
      93237.71c -1.88297594525e-07
      55135.71c -1.12590294487e-05
      94240.71c -1.12433979214e-06
      94241.71c -4.40461364834e-08
      94242.71c -6.28989409209e-10
c ******************************************************************************
c graphite rho = 1.6   g/cm^3
c S(a,b)- Carbon @ 300K
m2    6000.70c             -1  $MAT
c  Air rho = 0.001239 $ RSR AIR
m3 6000.70c -0.000124 7014.70c -0.755268 8016.70c -0.231781 18000 -0.012827
c Light water rho=1.00 g/cm^3
c S(a,b)- H20 @ 300K
m6    1001.70c        0.66667  $ Water
      8016.70c        0.33333
c Aluminum rho = 2.699 g/cm^3
m7    13027.70c            -1  $ Aluminum, 2.699 g/cc
c Zirconium $ Zirc  Filler, 6.5 g/cc
c S(a,b)- H20 @ 300K
m8    40090.71c  -1
c Stainless Steel rho =  7.9 g/cm^3
c m9  26000.55c   -1.0
m9    26000.55c    -0.67  $MAT
      24000.50c    -0.17
      28000.50c    -0.08849997
      42096.70c    -0.004452
c B4C SHIM & REG rod rho > 2.48 g/cm^3, email from GA,
c NEED TO ADJUST FOR BURNUP TO DIFFERENTIATE BETWEEN NEW ROD??
c rho 1.60-1.75 g/cm^3 from B. Ma
c 16% Boron from UIC TRIGA SAR
m11   6000.70c          -0.84  $ Pulse Rod, 1.6 g/cc,
      5010.70c          -0.0296
      5011.70c          -0.1304
c B4C rho > 2.48 g/cc, SHIM & REG, BUT THESE ARE OLD & PART BURNED
m12   6000.70c          -0.22  $ Shim Rod, 2.5 g/cc
      5010.70c          -0.1443
      5011.70c          -0.6357
c B4C rho >2.48 g/cc, SAFETY ROD
m13   6000.70c          -0.22  $ Shim Rod, 2.5 g/cc
      5010.70c          -0.1443
      5011.70c          -0.6357
c    ***********************************************************
c    borated polyethylene;   density= 1.000000 g/cm3
c    ***********************************************************
m5     1001     -0.125355
       5010.70c -0.100000
       6000     -0.774645 
c    ***********************************************************
c    boral               ;   density= 2.699 g/cm3
c    ***********************************************************
m14     1001     -0.1
       13027.70c -0.9
c    ***********************************************************
c    concrete            ;   density= 2.4 g/cm3
c    ***********************************************************
m15     1001.60c    -0.000431876637963
        1002.60c    -4.96715255912e-08
        8016.60c    -0.315386343063
        13027.60c    -0.0462515554425
        14028.62c    -0.083255735972
        14029.62c    -0.00422752391804
        14030.62c    -0.0027867246941
        16032.70c    -0.0902699845841
        16033.70c    -0.0260889700488
        16034.70c    -0.000208865661404
        16036.70c    -0.00117899169398
        20040.70c    -5.49646477379e-06
        20042.70c    -0.457866051098
        20043.70c    -0.00305587249007
        20044.70c    -0.000637624089892
        20046.70c    -0.00985247297418
        20048.70c    -1.88925656264e-05
        26054.60c    -0.000883227443035
        26056.62c    -0.00279763478884
        26057.60c    -0.0439168832191
        26058.60c    -0.00101423235544
c    ***********************************************************
c    borated concrete            ;   density= 2.4 g/cm3
c    ***********************************************************
m16     1001.60c    -0.000429664840336
        1002.60c    -4.94171396097e-08
        5010.60c    -0.00512136437429
        8016.60c    -0.313771134682
        13027.60c    -0.0460146843742
        14028.62c    -0.0828293530118
        14029.62c    -0.00420587322766
        14030.62c    -0.00277245286153
        16032.70c    -0.089807679101
        16033.70c    -0.025955358927
        16034.70c    -0.000207795984247
        16036.70c    -0.00117295364792
        20040.70c    -5.46831537492e-06
        20042.70c    -0.455521152215
        20043.70c    -0.00304022225357
        20044.70c    -0.000634358584594
        20046.70c    -0.00980201487009
        20048.70c    -1.87958099139e-05
        26054.60c    -0.000878704113474
        26056.62c    -0.0027833070817
        26057.60c    -0.0436919688579
        26058.60c    -0.00100903810199
c    ***********************************************************
c    lead               ;   density= 11.34 g/cm3
c    ***********************************************************
m17     082208.70c     -1
c ******************************************************************************
imp:n             0            1 1680r          $ 1, 63012
c ******************************************************************************
c Volume of each cell
VOL NO         0         0   23186.1   7981.87         0         0         0 &
               0    119806   2876.05    23.606 0 0 0 0 0 0       0         0 &
         27.2206   263.299         0         0         0   292.266         0 &
               0         0    37.203   360.755         0         0         0 &
         23.4395   153.542         0         0         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0   90.5361         0   6.24953 &
         90.5361   26.3186         0         0         0 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252         0         0 &
               0   90.5361         0   6.24953   90.5361   26.3186         0 &
               0         0 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252         0         0         0   90.5361         0 &
         6.24953   90.5361   26.3186         0         0         0 55.504252 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252         0 &
               0         0   90.5361         0   6.24953   90.5361   26.3186 &
               0         0         0 55.504252 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252         0         0         0   90.5361 &
               0   6.24953   90.5361   26.3186         0         0         0 &
       55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
               0         0         0   90.5361         0   6.24953   90.5361 &
         26.3186         0         0         0 55.504252 55.504252 55.504252 &
       55.504252 55.504252 55.504252 55.504252         0         0         0 &
         90.5361         0   6.24953   90.5361   26.3186         0         0 &
               0 55.504252 55.504252 55.504252 55.504252 55.504252 55.504252 &
       55.504252         0         0         0         0         0         0 &
               0    602.17         0         0         0         0         0 &
               0    18.099   4.87847         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0         0         0         0 &
               0         0         0         0 
c ******************************************************************************
NPS 2.5E11
NONU
c ******************************************************************************
c                             SOURCE SPECIFICATION                              
c ******************************************************************************
SDEF ERG=D1 RAD=D2  AXS=0 0 1  POS=D3  EXT=FPOS=D5 
SP1 -3
SI2   0  1.8669
SP2 -21  1
SI3  L     3.121883   2.586013  0.0000       3.800495  -1.410623  0.0000
           0.678612  -3.996636  0.0000      -3.121883  -2.586013  0.0000
          -3.800495   1.410623  0.0000      -0.678612   3.996636  0.0000
           6.145963   5.091011  0.0000       7.868066   1.335964  0.0000
           7.481927  -2.777054  0.0000       5.091011  -6.145963  0.0000
           1.335964  -7.868066  0.0000      -2.777054  -7.481927  0.0000
          -7.868066  -1.335964  0.0000      -7.481927   2.777054  0.0000
          -5.091011   6.145963  0.0000      -1.335964   7.868066  0.0000
           2.777054   7.481927  0.0000       9.199383   7.620314  0.0000
          11.250894   4.014378  0.0000      11.945380  -0.075751  0.0000
           9.102000  -7.736370  0.0000       5.907088  10.382877  0.0000
           1.999694  11.777056  0.0000      -2.148893  11.750749  0.0000
          -6.038292  10.307127  0.0000      -9.199383  -7.620314  0.0000
          11.250894  -4.014378  0.0000      11.945380   0.075751  0.0000
          11.199077   4.156742  0.0000      -9.102000   7.736370  0.0000
          -5.907088  10.382877  0.0000       2.148893  11.750749  0.0000
           6.038292  10.307127  0.0000      14.466832   6.634635  0.0000
          15.691056   2.664274  0.0000      15.845961  -1.487653  0.0000
          14.920990  -5.538199  0.0000      12.979179  -9.211326  0.0000
          10.152857  12.256716  0.0000       6.634635  14.466832  0.0000
           2.664274  15.691056  0.0000      -1.487653  15.845961  0.0000
          -5.538199  14.920990  0.0000      -9.211326  12.979179  0.0000
          12.256716  10.152857  0.0000      14.466832  -6.634635  0.0000
          15.691056  -2.664274  0.0000      15.845961   1.487653  0.0000
          14.920990   5.538199  0.0000      12.979179   9.211326  0.0000
          10.152857  12.256716  0.0000      -6.634635  14.466832  0.0000
          -2.664274  15.691056  0.0000       1.487653  15.845961  0.0000
           5.538199  14.920990  0.0000       9.211326  12.979179  0.0000
          15.316005  12.687021  0.0000      17.619093   9.225402  0.0000
          19.152143   5.360590  0.0000      19.848152   1.261493  0.0000
          19.676701  -2.892736  0.0000      18.645284  -6.920539  0.0000
          16.798979  10.645881  0.0000      14.218478  13.905947  0.0000
          11.016561  16.558257  0.0000       3.329279  19.607559  0.0000
          -0.820114  19.871283  0.0000      -4.933665  19.266538  0.0000
          -8.831591  17.819751  0.0000      12.343534  15.594155  0.0000
          15.316005  12.687021  0.0000      17.619093  -9.225402  0.0000
          19.152143  -5.360590  0.0000      19.848152  -1.261493  0.0000
          19.676701   2.892736  0.0000      18.645284   6.920539  0.0000
          16.798979  10.645881  0.0000      14.218478  13.905947  0.0000
          11.016561  16.558257  0.0000      -7.333167  18.486891  0.0000
          -3.329279  19.607559  0.0000       0.820114  19.871283  0.0000
           4.933665  19.266538  0.0000       8.831591  17.819751  0.0000
          12.343534  15.594155  0.0000
SP3       12.888540  12.818680  14.034930  16.044670  14.361910
          13.134620  10.707451  10.735753   9.720790  11.764750
          12.813460  15.621980  15.536000  13.653520  12.730190
          10.678500  11.040779   8.568636   8.675308   7.567440
           8.680720  10.396044  11.619410  12.568620  13.445340
          14.135010  13.741520  13.221730  12.652750  11.766770
           9.878205   8.565274   8.971678   6.211484   6.598888
           6.070626   5.763332   6.928482   8.047083   8.967637
           9.781396  10.291277  10.675300  10.733110  10.657730
          11.158060  10.861130  11.088050  10.384870   9.839594
           9.111963   7.926564   6.858617   6.996310   7.116448
           6.615234   4.919071   4.959257   4.990899   4.967203
           4.948176   4.952617   5.716737   5.992301   6.981035
           7.885199   7.683075   8.428431   8.900708   8.805337
           9.179130   9.338846   9.383957   8.968991   8.515944
           8.713837   8.015738   7.973371   7.470658   6.811665
           5.938480   6.101704   5.875097   5.474106   5.056876
c * added by ADVANTG
sb3    3.71542e-03 5.27673e-03 2.16504e-03 1.30232e-03 1.27093e-03 1.39998e-03
       4.19318e-03 1.07425e-02 7.36844e-03 3.51726e-03 1.32482e-03 7.73616e-04
       5.60226e-04 4.03455e-04 4.29229e-04 5.98216e-04 1.55184e-03 4.07729e-03
       1.24620e-02 2.17361e-02 4.19359e-03 1.84894e-03 8.85433e-04 5.09796e-04
       2.83417e-04 2.23136e-04 2.27678e-02 4.14305e-02 1.79145e-02 1.60861e-04
       1.96369e-04 6.22427e-04 1.60661e-03 8.94118e-03 3.16680e-02 3.98619e-02
       1.13387e-02 4.80099e-03 2.46353e-03 1.20233e-03 6.11648e-04 3.21169e-04
       1.85009e-04 1.13260e-04 6.28368e-03 1.74903e-02 5.47844e-02 7.65218e-02
       2.22534e-02 7.25276e-03 2.78971e-03 1.15001e-04 1.63671e-04 3.29237e-04
       7.24256e-04 1.55721e-03 3.21803e-03 7.18837e-03 2.25613e-02 8.27386e-02
       4.77817e-02 1.37877e-02 5.98009e-03 3.13022e-03 1.89548e-03 5.35595e-04
       2.78512e-04 1.68112e-04 1.01884e-04 3.05280e-03 6.37352e-03 1.40702e-02
       4.32573e-02 1.49359e-01 8.25857e-02 2.43081e-02 8.44921e-03 4.20424e-03
       2.01773e-03 9.29497e-05 1.42264e-04 2.80289e-04 5.23650e-04 9.53103e-04
       1.67898e-03
DS5  S   100  101  102  103  104
         105  106  107  108  109
         110  111  112  113  114
         115  116  117  118  119
         120  121  122  123  124
         125  126  127  128  129
         130  131  132  133  134
         135  136  137  138  139
         140  141  142  143  144
         145  146  147  148  149
         150  151  152  153  154
         155  156  157  158  159
         160  161  162  163  164
         165  166  167  168  169
         170  171  172  173  174
         175  176  177  178  179
         180  181  182  183  184
SI100  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP100  D     0.000000   1.240810   1.780490   2.202120
             2.296010   2.148310   1.852160   1.368640
c * added by ADVANTG
sb100  0.00000e+00 1.08111e-01 2.71604e-01 3.34707e-01 1.87772e-01 6.61194e-02
       2.34349e-02 8.25226e-03
SI101  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP101  D     0.000000   1.248720   1.771480   2.196870
             2.260620   2.136830   1.841560   1.362600
c * added by ADVANTG
sb101  0.00000e+00 1.04041e-01 2.78339e-01 3.62685e-01 1.74040e-01 5.55285e-02
       1.89658e-02 6.40124e-03
SI102  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP102  D     0.000000   1.388750   1.978190   2.423320
             2.491730   2.324300   1.984590   1.444050
c * added by ADVANTG
sb102  0.00000e+00 1.18275e-01 2.60469e-01 3.18624e-01 1.87356e-01 7.56839e-02
       2.94060e-02 1.01873e-02
SI103  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP103  D     0.000000   1.584330   2.273060   2.775340
             2.860830   2.670350   2.246720   1.634040
c * added by ADVANTG
sb103  0.00000e+00 1.15770e-01 2.63759e-01 3.14999e-01 1.87001e-01 7.70562e-02
       3.02373e-02 1.11778e-02
SI104  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP104  D     0.000000   1.409150   2.032350   2.472330
             2.548990   2.389360   2.023750   1.485980
c * added by ADVANTG
sb104  0.00000e+00 1.12750e-01 2.66065e-01 3.33430e-01 1.83812e-01 6.69939e-02
       2.72309e-02 9.71741e-03
SI105  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP105  D     0.000000   1.275460   1.816140   2.220230
             2.319010   2.193950   1.899810   1.410020
c * added by ADVANTG
sb105  0.00000e+00 1.16952e-01 2.56435e-01 3.06192e-01 1.92788e-01 8.22510e-02
       3.32401e-02 1.21410e-02
SI106  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP106  D     0.000000   0.983671   1.416290   1.724620
             1.882580   1.860050   1.616180   1.224060
c * added by ADVANTG
sb106  0.00000e+00 1.05486e-01 2.67355e-01 3.27652e-01 1.88109e-01 7.45585e-02
       2.69246e-02 9.91552e-03
SI107  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP107  D     0.000000   0.983973   1.412280   1.726010
             1.878320   1.851670   1.643000   1.240500
c * added by ADVANTG
sb107  0.00000e+00 9.38462e-02 2.97789e-01 3.66975e-01 1.70699e-01 4.96908e-02
       1.58971e-02 5.10359e-03
SI108  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP108  D     0.000000   0.860820   1.217580   1.502210
             1.620310   1.600660   1.609430   1.309780
c * added by ADVANTG
sb108  0.00000e+00 1.00693e-01 2.80963e-01 3.49996e-01 1.78783e-01 5.92868e-02
       2.23638e-02 7.91515e-03
SI109  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP109  D     0.000000   1.098190   1.558920   1.903540
             2.069680   2.020630   1.780920   1.332870
c * added by ADVANTG
sb109  0.00000e+00 1.10397e-01 2.60813e-01 3.12013e-01 1.89506e-01 8.45362e-02
       3.14590e-02 1.12765e-02
SI110  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP110  D     0.000000   1.209880   1.731920   2.124260
             2.299300   2.206310   1.878030   1.363760
c * added by ADVANTG
sb110  0.00000e+00 1.17091e-01 2.39847e-01 2.86977e-01 2.03932e-01 9.62310e-02
       4.04032e-02 1.55180e-02
SI111  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP111  D     0.000000   1.500680   2.143850   2.609500
             2.793960   2.686950   2.257940   1.629100
c * added by ADVANTG
sb111  0.00000e+00 1.25780e-01 2.35243e-01 2.78549e-01 1.95965e-01 1.02616e-01
       4.38377e-02 1.80092e-02
SI112  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP112  D     0.000000   1.503940   2.137100   2.585830
             2.771910   2.673230   2.249750   1.614240
c * added by ADVANTG
sb112  0.00000e+00 1.18801e-01 2.52180e-01 3.12805e-01 1.91951e-01 7.83837e-02
       3.27570e-02 1.31223e-02
SI113  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP113  D     0.000000   1.300860   1.843760   2.253190
             2.422590   2.359200   2.017790   1.456130
c * added by ADVANTG
sb113  0.00000e+00 1.22390e-01 2.43734e-01 2.92483e-01 1.90613e-01 9.41368e-02
       4.09042e-02 1.57383e-02
SI114  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP114  D     0.000000   1.172240   1.689450   2.069160
             2.229930   2.200640   1.936300   1.432470
c * added by ADVANTG
sb114  0.00000e+00 1.18370e-01 2.31659e-01 2.76450e-01 1.98377e-01 1.05591e-01
       4.95801e-02 1.99718e-02
SI115  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP115  D     0.000000   0.900680   1.295960   1.581080
             1.738270   1.868810   1.866140   1.427560
c * added by ADVANTG
sb115  0.00000e+00 1.14399e-01 2.19409e-01 2.62458e-01 1.98139e-01 1.18522e-01
       6.11385e-02 2.59342e-02
SI116  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP116  D     0.000000   0.984409   1.421010   1.766630
             1.932700   1.935320   1.716200   1.284510
c * added by ADVANTG
sb116  0.00000e+00 1.09316e-01 2.34654e-01 2.92573e-01 2.04079e-01 1.01078e-01
       4.17802e-02 1.65205e-02
SI117  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP117  D     0.000000   0.755796   1.075380   1.324760
             1.450130   1.458330   1.395070   1.109170
c * added by ADVANTG
sb117  0.00000e+00 1.03630e-01 2.48686e-01 3.10545e-01 1.95916e-01 8.82088e-02
       3.82061e-02 1.48080e-02
SI118  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP118  D     0.000000   0.797668   1.125550   1.371070
             1.519180   1.488820   1.346320   1.026700
c * added by ADVANTG
sb118  0.00000e+00 9.40371e-02 2.94286e-01 3.55855e-01 1.77145e-01 5.46523e-02
       1.78261e-02 6.19847e-03
SI119  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP119  D     0.000000   0.660159   0.927451   1.138370
             1.237210   1.251770   1.271860   1.080620
c * added by ADVANTG
sb119  0.00000e+00 8.27922e-02 3.15664e-01 3.97426e-01 1.53124e-01 3.49554e-02
       1.17478e-02 4.29116e-03
SI120  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP120  D     0.000000   0.766970   1.094760   1.338320
             1.431970   1.427790   1.432130   1.188780
c * added by ADVANTG
sb120  0.00000e+00 1.04364e-01 2.54091e-01 3.13883e-01 1.87904e-01 8.68778e-02
       3.77619e-02 1.51184e-02
SI121  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP121  D     0.000000   0.988404   1.388610   1.704140
             1.833380   1.790160   1.548360   1.142990
c * added by ADVANTG
sb121  0.00000e+00 1.13989e-01 2.24760e-01 2.83359e-01 2.01865e-01 1.07845e-01
       4.89237e-02 1.92578e-02
SI122  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP122  D     0.000000   1.105900   1.587170   1.919950
             2.085080   2.007870   1.683780   1.229660
c * added by ADVANTG
sb122  0.00000e+00 1.18469e-01 2.15332e-01 2.64120e-01 2.04028e-01 1.19475e-01
       5.49758e-02 2.36014e-02
SI123  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP123  D     0.000000   1.213130   1.721570   2.091460
             2.254170   2.163250   1.812860   1.312180
c * added by ADVANTG
sb123  0.00000e+00 1.17987e-01 2.10469e-01 2.52792e-01 2.07421e-01 1.25995e-01
       5.93149e-02 2.60208e-02
SI124  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP124  D     0.000000   1.303210   1.843230   2.256210
             2.427510   2.300350   1.916660   1.398170
c * added by ADVANTG
sb124  0.00000e+00 1.23460e-01 2.10848e-01 2.57288e-01 2.00231e-01 1.23000e-01
       5.84151e-02 2.67570e-02
SI125  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP125  D     0.000000   1.357880   1.951570   2.362290
             2.535050   2.426420   2.044120   1.457680
c * added by ADVANTG
sb125  0.00000e+00 1.23272e-01 2.18483e-01 2.69131e-01 1.98375e-01 1.13148e-01
       5.36638e-02 2.39282e-02
SI126  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP126  D     0.000000   1.314340   1.889450   2.300200
             2.458190   2.352090   1.991740   1.435510
c * added by ADVANTG
sb126  0.00000e+00 9.01488e-02 2.85940e-01 3.67147e-01 1.78872e-01 5.50708e-02
       1.74093e-02 5.41242e-03
SI127  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP127  D     0.000000   1.266950   1.801210   2.223460
             2.367540   2.271130   1.909470   1.381970
c * added by ADVANTG
sb127  0.00000e+00 8.47967e-02 3.16806e-01 4.00740e-01 1.52617e-01 3.30713e-02
       9.13953e-03 2.83024e-03
SI128  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP128  D     0.000000   1.200200   1.708320   2.107500
             2.270900   2.175250   1.847410   1.343170
c * added by ADVANTG
sb128  0.00000e+00 9.53911e-02 2.88956e-01 3.62109e-01 1.77794e-01 5.33754e-02
       1.68100e-02 5.56409e-03
SI129  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP129  D     0.000000   1.093100   1.560950   1.919380
             2.076840   2.054900   1.771850   1.289750
c * added by ADVANTG
sb129  0.00000e+00 1.28062e-01 2.07579e-01 2.51473e-01 2.03649e-01 1.21551e-01
       6.10566e-02 2.66288e-02
SI130  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP130  D     0.000000   0.842535   1.189060   1.460510
             1.602850   1.732130   1.733250   1.317870
c * added by ADVANTG
sb130  0.00000e+00 1.15437e-01 1.97053e-01 2.42458e-01 1.97838e-01 1.34195e-01
       7.70459e-02 3.59730e-02
SI131  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP131  D     0.000000   0.696884   0.993250   1.246930
             1.366000   1.498310   1.563870   1.200030
c * added by ADVANTG
sb131  0.00000e+00 1.07855e-01 2.01141e-01 2.51498e-01 2.01715e-01 1.30100e-01
       7.39531e-02 3.37376e-02
SI132  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP132  D     0.000000   0.806038   1.147420   1.419120
             1.552010   1.558360   1.409940   1.078790
c * added by ADVANTG
sb132  0.00000e+00 1.09822e-01 2.22998e-01 2.79249e-01 2.01570e-01 1.12346e-01
       5.22257e-02 2.17889e-02
SI133  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP133  D     0.000000   0.559501   0.778096   0.957472
             1.032080   1.035130   1.006880   0.842325
c * added by ADVANTG
sb133  0.00000e+00 9.65276e-02 2.76742e-01 3.43131e-01 1.77122e-01 6.71487e-02
       2.79514e-02 1.13779e-02
SI134  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP134  D     0.000000   0.620698   0.864578   1.049290
             1.138720   1.118170   1.002010   0.805422
c * added by ADVANTG
sb134  0.00000e+00 8.24363e-02 3.22571e-01 3.95960e-01 1.50355e-01 3.41611e-02
       1.06551e-02 3.86133e-03
SI135  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP135  D     0.000000   0.559296   0.782516   0.937998
             1.012590   1.001100   0.965929   0.811197
c * added by ADVANTG
sb135  0.00000e+00 7.53945e-02 3.37000e-01 4.11145e-01 1.38029e-01 2.66854e-02
       8.57733e-03 3.16810e-03
SI136  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP136  D     0.000000   0.495705   0.697040   0.850944
             0.911133   0.910926   0.999580   0.898004
c * added by ADVANTG
sb136  0.00000e+00 8.95267e-02 2.88913e-01 3.60591e-01 1.69191e-01 5.70999e-02
       2.44320e-02 1.02467e-02
SI137  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP137  D     0.000000   0.633242   0.896897   1.091350
             1.180260   1.148710   1.089880   0.888143
c * added by ADVANTG
sb137  0.00000e+00 1.05023e-01 2.44173e-01 3.03933e-01 1.96697e-01 9.14732e-02
       4.09334e-02 1.77675e-02
SI138  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP138  D     0.000000   0.757528   1.077210   1.310030
             1.417380   1.373930   1.201840   0.909165
c * added by ADVANTG
sb138  0.00000e+00 1.10699e-01 2.16394e-01 2.70104e-01 2.05327e-01 1.18248e-01
       5.58287e-02 2.33989e-02
SI139  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP139  D     0.000000   0.873329   1.214790   1.470180
             1.579080   1.535900   1.314520   0.979838
c * added by ADVANTG
sb139  0.00000e+00 1.16901e-01 2.11935e-01 2.47436e-01 2.05184e-01 1.25214e-01
       6.47763e-02 2.85539e-02
SI140  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP140  D     0.000000   0.953616   1.323850   1.612890
             1.763720   1.677750   1.408400   1.041170
c * added by ADVANTG
sb140  0.00000e+00 1.19015e-01 1.98694e-01 2.39979e-01 2.07112e-01 1.35241e-01
       6.75965e-02 3.23626e-02
SI141  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP141  D     0.000000   0.991547   1.389610   1.709210
             1.864890   1.776060   1.478560   1.081400
c * added by ADVANTG
sb141  0.00000e+00 1.19444e-01 1.98210e-01 2.36786e-01 2.03786e-01 1.37106e-01
       7.05108e-02 3.41566e-02
SI142  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP142  D     0.000000   1.050740   1.448210   1.786940
             1.924130   1.830350   1.527080   1.107850
c * added by ADVANTG
sb142  0.00000e+00 1.24166e-01 1.97484e-01 2.37422e-01 2.01828e-01 1.33498e-01
       7.08899e-02 3.47116e-02
SI143  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP143  D     0.000000   1.059530   1.480030   1.791990
             1.909750   1.826780   1.538160   1.126870
c * added by ADVANTG
sb143  0.00000e+00 1.29647e-01 1.96352e-01 2.38729e-01 2.02187e-01 1.29880e-01
       6.82528e-02 3.49525e-02
SI144  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP144  D     0.000000   1.041240   1.475110   1.782960
             1.903470   1.821120   1.521030   1.112800
c * added by ADVANTG
sb144  0.00000e+00 1.07338e-01 2.34220e-01 2.94065e-01 2.04632e-01 1.00540e-01
       4.26316e-02 1.65728e-02
SI145  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP145  D     0.000000   1.085020   1.524340   1.862290
             2.004110   1.923640   1.591860   1.166800
c * added by ADVANTG
sb145  0.00000e+00 9.59979e-02 2.79897e-01 3.45954e-01 1.80506e-01 6.67655e-02
       2.28489e-02 8.03045e-03
SI146  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP146  D     0.000000   1.061870   1.503500   1.823790
             1.926690   1.854020   1.555560   1.135700
c * added by ADVANTG
sb146  0.00000e+00 8.06838e-02 3.23035e-01 4.02228e-01 1.48263e-01 3.28870e-02
       9.71554e-03 3.18765e-03
SI147  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP147  D     0.000000   1.061120   1.514880   1.856740
             1.997570   1.900750   1.590090   1.166900
c * added by ADVANTG
sb147  0.00000e+00 7.72415e-02 3.33188e-01 4.13919e-01 1.39710e-01 2.62083e-02
       7.34308e-03 2.39000e-03
SI148  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP148  D     0.000000   1.009930   1.425390   1.711290
             1.843390   1.776640   1.509860   1.108370
c * added by ADVANTG
sb148  0.00000e+00 9.27243e-02 3.00547e-01 3.58379e-01 1.69142e-01 5.44570e-02
       1.84305e-02 6.32053e-03
SI149  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP149  D     0.000000   0.950904   1.309120   1.603870
             1.732910   1.704140   1.462490   1.076160
c * added by ADVANTG
sb149  0.00000e+00 1.04761e-01 2.40485e-01 3.02816e-01 1.98935e-01 9.74931e-02
       4.01215e-02 1.53885e-02
SI150  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP150  D     0.000000   0.850653   1.195280   1.468240
             1.585880   1.576580   1.393540   1.041790
c * added by ADVANTG
sb150  0.00000e+00 1.09733e-01 2.15690e-01 2.68118e-01 2.05815e-01 1.19155e-01
       5.72385e-02 2.42494e-02
SI151  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP151  D     0.000000   0.701169   0.982535   1.195040
             1.342920   1.385670   1.308830   1.010400
c * added by ADVANTG
sb151  0.00000e+00 1.23255e-01 1.85807e-01 2.19438e-01 2.00137e-01 1.41380e-01
       8.71938e-02 4.27881e-02
SI152  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP152  D     0.000000   0.543686   0.766733   0.934398
             1.037380   1.209110   1.314340   1.052970
c * added by ADVANTG
sb152  0.00000e+00 1.06831e-01 1.77456e-01 2.15185e-01 1.89707e-01 1.51645e-01
       1.03547e-01 5.56289e-02
SI153  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP153  D     0.000000   0.594471   0.845097   1.038660
             1.159450   1.225890   1.195440   0.937302
c * added by ADVANTG
sb153  0.00000e+00 1.10321e-01 1.87302e-01 2.26499e-01 1.97542e-01 1.44146e-01
       9.01489e-02 4.40423e-02
SI154  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP154  D     0.000000   0.632838   0.896384   1.125800
             1.209690   1.233100   1.134010   0.884626
c * added by ADVANTG
sb154  0.00000e+00 1.08444e-01 1.96609e-01 2.47405e-01 2.03502e-01 1.34455e-01
       7.38112e-02 3.57744e-02
SI155  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP155  D     0.000000   0.578956   0.821260   1.011470
             1.092560   1.111740   1.101500   0.897748
c * added by ADVANTG
sb155  0.00000e+00 1.07644e-01 2.06919e-01 2.58721e-01 2.00494e-01 1.24620e-01
       6.92227e-02 3.23796e-02
SI156  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP156  D     0.000000   0.442761   0.605486   0.729371
             0.797214   0.796234   0.821341   0.726664
c * added by ADVANTG
sb156  0.00000e+00 1.06571e-01 2.05324e-01 2.52168e-01 2.02206e-01 1.25114e-01
       7.27400e-02 3.58761e-02
SI157  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP157  D     0.000000   0.481972   0.643905   0.772660
             0.827283   0.816610   0.765431   0.651396
c * added by ADVANTG
sb157  0.00000e+00 1.08323e-01 2.43083e-01 2.95027e-01 1.91535e-01 9.64092e-02
       4.55255e-02 2.00969e-02
SI158  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP158  D     0.000000   0.511810   0.681525   0.817120
             0.847381   0.804800   0.722670   0.605593
c * added by ADVANTG
sb158  0.00000e+00 8.85670e-02 3.12513e-01 3.78890e-01 1.48613e-01 4.62996e-02
       1.78702e-02 7.24665e-03
SI159  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP159  D     0.000000   0.522488   0.702226   0.829266
             0.833380   0.786053   0.702806   0.590984
c * added by ADVANTG
sb159  0.00000e+00 6.84299e-02 3.64555e-01 4.34431e-01 1.08446e-01 1.65019e-02
       5.48636e-03 2.14936e-03
SI160  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP160  D     0.000000   0.525588   0.684724   0.814284
             0.832223   0.778416   0.700895   0.612046
c * added by ADVANTG
sb160  0.00000e+00 7.81243e-02 3.43498e-01 4.12956e-01 1.26667e-01 2.58947e-02
       9.08837e-03 3.77197e-03
SI161  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP161  D     0.000000   0.514878   0.668464   0.796685
             0.835203   0.792800   0.726601   0.617986
c * added by ADVANTG
sb161  0.00000e+00 1.00393e-01 2.85119e-01 3.43419e-01 1.67041e-01 6.51389e-02
       2.75454e-02 1.13435e-02
SI162  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP162  D     0.000000   0.566898   0.750696   0.924056
             0.987414   0.949416   0.840948   0.697309
c * added by ADVANTG
sb162  0.00000e+00 1.11727e-01 2.26898e-01 2.79101e-01 2.01134e-01 1.07858e-01
       5.10311e-02 2.22505e-02
SI163  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP163  D     0.000000   0.608311   0.804952   0.957928
             1.032240   1.003890   0.884295   0.700685
c * added by ADVANTG
sb163  0.00000e+00 1.14370e-01 2.06766e-01 2.51731e-01 2.02169e-01 1.28987e-01
       6.64184e-02 2.95582e-02
SI164  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP164  D     0.000000   0.712566   0.939110   1.126320
             1.220220   1.173220   1.009710   0.799889
c * added by ADVANTG
sb164  0.00000e+00 1.15189e-01 1.95999e-01 2.41521e-01 2.04583e-01 1.35785e-01
       7.27219e-02 3.42018e-02
SI165  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP165  D     0.000000   0.801194   1.069120   1.278120
             1.373320   1.328310   1.150900   0.884235
c * added by ADVANTG
sb165  0.00000e+00 1.20684e-01 1.87554e-01 2.21118e-01 2.04402e-01 1.43352e-01
       8.28994e-02 3.99901e-02
SI166  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP166  D     0.000000   0.781037   1.039630   1.250060
             1.357440   1.297100   1.104940   0.852868
c * added by ADVANTG
sb166  0.00000e+00 1.19859e-01 1.83812e-01 2.23732e-01 2.02381e-01 1.44029e-01
       8.24836e-02 4.37035e-02
SI167  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP167  D     0.000000   0.854946   1.149160   1.386230
             1.486630   1.421120   1.213700   0.916645
c * added by ADVANTG
sb167  0.00000e+00 1.20905e-01 1.80143e-01 2.26670e-01 2.03885e-01 1.43408e-01
       8.15020e-02 4.34862e-02
SI168  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP168  D     0.000000   0.889170   1.205360   1.481720
             1.579610   1.499570   1.278510   0.966768
c * added by ADVANTG
sb168  0.00000e+00 1.22679e-01 1.87330e-01 2.21501e-01 2.02235e-01 1.41488e-01
       8.18722e-02 4.28952e-02
SI169  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP169  D     0.000000   0.896301   1.209310   1.454140
             1.550150   1.476340   1.259320   0.959776
c * added by ADVANTG
sb169  0.00000e+00 1.14299e-01 2.00316e-01 2.48302e-01 2.05659e-01 1.32366e-01
       6.83692e-02 3.06901e-02
SI170  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP170  D     0.000000   0.924010   1.250790   1.508940
             1.614060   1.558530   1.318160   1.004640
c * added by ADVANTG
sb170  0.00000e+00 1.11657e-01 2.14725e-01 2.63984e-01 2.04222e-01 1.21744e-01
       5.88709e-02 2.47977e-02
SI171  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP171  D     0.000000   0.949956   1.275000   1.532850
             1.647220   1.586100   1.335570   1.012150
c * added by ADVANTG
sb171  0.00000e+00 1.08412e-01 2.44229e-01 2.98965e-01 1.96180e-01 9.56120e-02
       4.05504e-02 1.60523e-02
SI172  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP172  D     0.000000   0.962427   1.281950   1.560570
             1.640860   1.575270   1.348560   1.014320
c * added by ADVANTG
sb172  0.00000e+00 8.54370e-02 3.08867e-01 3.84002e-01 1.49932e-01 4.80224e-02
       1.74603e-02 6.28044e-03
SI173  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP173  D     0.000000   0.915217   1.239670   1.465850
             1.564670   1.512200   1.280340   0.991044
c * added by ADVANTG
sb173  0.00000e+00 6.75398e-02 3.62490e-01 4.29779e-01 1.14503e-01 1.80141e-02
       5.62332e-03 2.05062e-03
SI174  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP174  D     0.000000   0.863225   1.160890   1.389150
             1.471810   1.435740   1.239090   0.956039
c * added by ADVANTG
sb174  0.00000e+00 7.52625e-02 3.40462e-01 4.13843e-01 1.29831e-01 2.76378e-02
       9.47497e-03 3.48830e-03
SI175  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP175  D     0.000000   0.880498   1.190390   1.416150
             1.522270   1.474850   1.271410   0.958269
c * added by ADVANTG
sb175  0.00000e+00 9.59691e-02 2.87349e-01 3.41111e-01 1.70001e-01 6.81337e-02
       2.71691e-02 1.02675e-02
SI176  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP176  D     0.000000   0.812146   1.084780   1.287720
             1.395390   1.348270   1.179330   0.908102
c * added by ADVANTG
sb176  0.00000e+00 1.12885e-01 2.26103e-01 2.76522e-01 2.04403e-01 1.07860e-01
       5.13101e-02 2.09172e-02
SI177  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP177  D     0.000000   0.770964   1.060600   1.298250
             1.390960   1.358230   1.185200   0.909167
c * added by ADVANTG
sb177  0.00000e+00 1.08512e-01 2.01917e-01 2.57037e-01 2.06612e-01 1.29743e-01
       6.69571e-02 2.92228e-02
SI178  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP178  D     0.000000   0.708755   0.973808   1.198260
             1.311660   1.276690   1.124120   0.877365
c * added by ADVANTG
sb178  0.00000e+00 1.08825e-01 1.91563e-01 2.38024e-01 2.09782e-01 1.39422e-01
       7.71897e-02 3.51938e-02
SI179  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP179  D     0.000000   0.640860   0.871099   1.074880
             1.168390   1.181020   1.047510   0.827906
c * added by ADVANTG
sb179  0.00000e+00 1.16583e-01 1.75776e-01 2.15565e-01 1.99455e-01 1.49057e-01
       9.38851e-02 4.96779e-02
SI180  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP180  D     0.000000   0.564760   0.755948   0.920831
             0.996073   1.020310   0.929015   0.751543
c * added by ADVANTG
sb180  0.00000e+00 1.13799e-01 1.76701e-01 2.16210e-01 1.94491e-01 1.50924e-01
       9.53762e-02 5.24991e-02
SI181  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP181  D     0.000000   0.573355   0.778829   0.948527
             1.047370   1.050320   0.942398   0.760905
c * added by ADVANTG
sb181  0.00000e+00 1.09952e-01 1.80028e-01 2.22605e-01 2.02200e-01 1.47220e-01
       9.04330e-02 4.75615e-02
SI182  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP182  D     0.000000   0.558104   0.752273   0.906638
             0.998475   1.011100   0.911747   0.736760
c * added by ADVANTG
sb182  0.00000e+00 1.11427e-01 1.85139e-01 2.23781e-01 2.00832e-01 1.46729e-01
       8.77700e-02 4.43217e-02
SI183  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP183  D     0.000000   0.519988   0.705394   0.842667
             0.926174   0.940647   0.851495   0.687741
c * added by ADVANTG
sb183  0.00000e+00 1.10743e-01 1.87474e-01 2.29306e-01 2.00110e-01 1.46462e-01
       8.41081e-02 4.17964e-02
SI184  H   -19.051000 -13.607857  -8.164714  -2.721571
             2.721571   8.164714  13.607857  19.051000
SP184  D     0.000000   0.481916   0.630222   0.772050
             0.837720   0.843882   0.809008   0.682078
c * added by ADVANTG
sb184  0.00000e+00 1.09871e-01 1.90697e-01 2.42965e-01 2.03149e-01 1.34933e-01
       7.89047e-02 3.94809e-02
c ******************************************************************************
mt2     grph.01T
mt6     lwtr.01
c ************************* TALLY SPECIFICATION ********************************
f7:n  (201100 201110 201120 201130 201140 201150 201160 202100 202110 202120 &
      202130 202140 202150 202160 203100 203110 203120 203130 203140 203150 &
      203160 204100 204110 204120 204130 204140 204150 204160 205100 205110 &
      205120 205130 205140 205150 205160 206100 206110 206120 206130 206140 &
      206150 206160 301100 301110 301120 301130 301140 301150 301160 302100 &
      302110 302120 302130 302140 302150 302160 303100 303110 303120 303130 &
      303140 303150 303160 304100 304110 304120 304130 304140 304150 304160 &
      305100 305110 305120 305130 305140 305150 305160 306100 306110 306120 &
      306130 306140 306150 306160 308100 308110 308120 308130 308140 308150 &
      308160 309100 309110 309120 309130 309140 309150 309160 310100 310110 &
      310120 310130 310140 310150 310160 311100 311110 311120 311130 311140 &
      311150 311160 312100 312110 312120 312130 312140 312150 312160 401100 &
      401110 401120 401130 401140 401150 401160 402100 402110 402120 402130 &
      402140 402150 402160 403100 403110 403120 403130 403140 403150 403160 &
      405100 405110 405120 405130 405140 405150 405160 406100 406110 406120 &
      406130 406140 406150 406160 407100 407110 407120 407130 407140 407150 &
      407160 408100 408110 408120 408130 408140 408150 408160 409100 409110 &
      409120 409130 409140 409150 409160 410100 410110 410120 410130 410140 &
      410150 410160 411100 411110 411120 411130 411140 411150 411160 412100 &
      412110 412120 412130 412140 412150 412160 413100 413110 413120 413130 &
      413140 413150 413160 414100 414110 414120 414130 414140 414150 414160 &
      415100 415110 415120 415130 415140 415150 415160 417100 417110 417120 &
      417130 417140 417150 417160 418100 418110 418120 418130 418140 418150 &
      418160 502100 502110 502120 502130 502140 502150 502160 503100 503110 &
      503120 503130 503140 503150 503160 504100 504110 504120 504130 504140 &
      504150 504160 505100 505110 505120 505130 505140 505150 505160 506100 &
      506110 506120 506130 506140 506150 506160 507100 507110 507120 507130 &
      507140 507150 507160 508100 508110 508120 508130 508140 508150 508160 &
      509100 509110 509120 509130 509140 509150 509160 510100 510110 510120 &
      510130 510140 510150 510160 511100 511110 511120 511130 511140 511150 &
      511160 512100 512110 512120 512130 512140 512150 512160 513100 513110 &
      513120 513130 513140 513150 513160 514100 514110 514120 514130 514140 &
      514150 514160 515100 515110 515120 515130 515140 515150 515160 516100 &
      516110 516120 516130 516140 516150 516160 517100 517110 517120 517130 &
      517140 517150 517160 518100 518110 518120 518130 518140 518150 518160 &
      519100 519110 519120 519130 519140 519150 519160 520100 520110 520120 &
      520130 520140 520150 520160 521100 521110 521120 521130 521140 521150 &
      521160 522100 522110 522120 522130 522140 522150 522160 523100 523110 &
      523120 523130 523140 523150 523160 524100 524110 524120 524130 524140 &
      524150 524160 601100 601110 601120 601130 601140 601150 601160 602100 &
      602110 602120 602130 602140 602150 602160 603100 603110 603120 603130 &
      603140 603150 603160 604100 604110 604120 604130 604140 604150 604160 &
      605100 605110 605120 605130 605140 605150 605160 606100 606110 606120 &
      606130 606140 606150 606160 607100 607110 607120 607130 607140 607150 &
      607160 608100 608110 608120 608130 608140 608150 608160 609100 609110 &
      609120 609130 609140 609150 609160 611100 611110 611120 611130 611140 &
      611150 611160 612100 612110 612120 612130 612140 612150 612160 613100 &
      613110 613120 613130 613140 613150 613160 614100 614110 614120 614130 &
      614140 614150 614160 615100 615110 615120 615130 615140 615150 615160 &
      616100 616110 616120 616130 616140 616150 616160 617100 617110 617120 &
      617130 617140 617150 617160 618100 618110 618120 618130 618140 618150 &
      618160 619100 619110 619120 619130 619140 619150 619160 620100 620110 &
      620120 620130 620140 620150 620160 621100 621110 621120 621130 621140 &
      621150 621160 622100 622110 622120 622130 622140 622150 622160 623100 &
      623110 623120 623130 623140 623150 623160 624100 624110 624120 624130 &
      624140 624150 624160 625100 625110 625120 625130 625140 625150 625160 &
      626100 626110 626120 626130 626140 626150 626160 627100 627110 627120 &
      627130 627140 627150 627160 628100 628110 628120 628130 628140 628150 &
      628160 629100 629110 629120 629130 629140 629150 629160 630100 630110 &
      630120 630130 630140 630150 630160) 
c
c
c
F11:N 115
FS11 46 
E11   1.000000e-11  4.000000e-09  1.000000e-08  2.530000e-08  4.000000e-08  
      5.000000e-08  6.000000e-08  8.000000e-08  1.000000e-07  1.500000e-07  
      2.000000e-07  2.500000e-07  3.250000e-07  3.500000e-07  3.750000e-07  
      4.500000e-07  6.250000e-07  1.010000e-06  1.080000e-06  1.130000e-06  
      5.000000e-06  6.250000e-06  6.500000e-06  6.880000e-06  7.000000e-06  
      2.050000e-05  2.120000e-05  2.180000e-05  3.600000e-05  3.710000e-05  
      6.500000e-05  6.750000e-05  1.010000e-04  1.050000e-04  1.160000e-04  
      1.180000e-04  1.880000e-04  1.920000e-04  2.250000e-03  3.740000e-03  
      1.700000e-02  2.000000e-02  5.000000e-02  2.000000e-01  2.700000e-01  
      3.300000e-01  4.700000e-01  6.000000e-01  7.500000e-01  8.610000e-01  
      1.200000e+00  1.500000e+00  1.850000e+00  3.000000e+00  4.300000e+00  
      6.430000e+00  2.000000e+01 
c
c
c
F21:N 115
FS21 46
E21   1.000000e-11  4.000000e-09  1.000000e-08  2.530000e-08  4.000000e-08  
      5.000000e-08  6.000000e-08  8.000000e-08  1.000000e-07  1.500000e-07  
      2.000000e-07  2.500000e-07  3.250000e-07  3.500000e-07  3.750000e-07  
      4.500000e-07  6.250000e-07  1.010000e-06  1.080000e-06  1.130000e-06  
      5.000000e-06  6.250000e-06  6.500000e-06  6.880000e-06  7.000000e-06  
      2.050000e-05  2.120000e-05  2.180000e-05  3.600000e-05  3.710000e-05  
      6.500000e-05  6.750000e-05  1.010000e-04  1.050000e-04  1.160000e-04  
      1.180000e-04  1.880000e-04  1.920000e-04  2.250000e-03  3.740000e-03  
      1.700000e-02  2.000000e-02  5.000000e-02  2.000000e-01  2.700000e-01  
      3.300000e-01  4.700000e-01  6.000000e-01  7.500000e-01  8.610000e-01  
      1.200000e+00  1.500000e+00  1.850000e+00  3.000000e+00  4.300000e+00  
      6.430000e+00  2.000000e+01 
C21                 0.000000e+00  1.736482e-01  3.420201e-01  5.000000e-01  
      6.427876e-01  7.660444e-01  8.660254e-01  9.396926e-01  9.848078e-01  
      9.961947e-01  9.993908e-01  9.998477e-01  9.999619e-01  1.000000e+00 T
c
F41:N 115
FS41 46
E41   1.000000e-11  4.000000e-09  1.000000e-08  2.530000e-08  4.000000e-08  
      5.000000e-08  6.000000e-08  8.000000e-08  1.000000e-07  1.500000e-07  
      2.000000e-07  2.500000e-07  3.250000e-07  3.500000e-07  3.750000e-07  
      4.500000e-07  6.250000e-07  1.010000e-06  1.080000e-06  1.130000e-06  
      5.000000e-06  6.250000e-06  6.500000e-06  6.880000e-06  7.000000e-06  
      2.050000e-05  2.120000e-05  2.180000e-05  3.600000e-05  3.710000e-05  
      6.500000e-05  6.750000e-05  1.010000e-04  1.050000e-04  1.160000e-04  
      1.180000e-04  1.880000e-04  1.920000e-04  2.250000e-03  3.740000e-03  
      1.700000e-02  2.000000e-02  5.000000e-02  2.000000e-01  2.700000e-01  
      3.300000e-01  4.700000e-01  6.000000e-01  7.500000e-01  8.610000e-01  
      1.200000e+00  1.500000e+00  1.850000e+00  3.000000e+00  4.300000e+00  
      6.430000e+00  2.000000e+01 
C41                 0.000000e+00  1.736482e-01  3.420201e-01  5.000000e-01  
      6.427876e-01  7.660444e-01  8.660254e-01  9.396926e-01  9.848078e-01  
      9.961947e-01  9.993908e-01  9.998477e-01  1.000000e+00 T
c
F51:N 115
FS51 46
E51   1.000000e-11  4.000000e-09  1.000000e-08  2.530000e-08  4.000000e-08  
      5.000000e-08  6.000000e-08  8.000000e-08  1.000000e-07  1.500000e-07  
      2.000000e-07  2.500000e-07  3.250000e-07  3.500000e-07  3.750000e-07  
      4.500000e-07  6.250000e-07  1.010000e-06  1.080000e-06  1.130000e-06  
      5.000000e-06  6.250000e-06  6.500000e-06  6.880000e-06  7.000000e-06  
      2.050000e-05  2.120000e-05  2.180000e-05  3.600000e-05  3.710000e-05  
      6.500000e-05  6.750000e-05  1.010000e-04  1.050000e-04  1.160000e-04  
      1.180000e-04  1.880000e-04  1.920000e-04  2.250000e-03  3.740000e-03  
      1.700000e-02  2.000000e-02  5.000000e-02  2.000000e-01  2.700000e-01  
      3.300000e-01  4.700000e-01  6.000000e-01  7.500000e-01  8.610000e-01  
      1.200000e+00  1.500000e+00  1.850000e+00  3.000000e+00  4.300000e+00  
      6.430000e+00  2.000000e+01 
C51                 0.000000e+00  1.736482e-01  3.420201e-01  5.000000e-01  
      6.427876e-01  7.660444e-01  8.660254e-01  9.396926e-01  9.848078e-01  
      9.961947e-01  9.998477e-01  1.000000e+00 T
c
c
F61:N 115
FS61 46
E61   1.000000e-11  4.000000e-09  1.000000e-08  2.530000e-08  4.000000e-08  
      5.000000e-08  6.000000e-08  8.000000e-08  1.000000e-07  1.500000e-07  
      2.000000e-07  2.500000e-07  3.250000e-07  3.500000e-07  3.750000e-07  
      4.500000e-07  6.250000e-07  1.010000e-06  1.080000e-06  1.130000e-06  
      5.000000e-06  6.250000e-06  6.500000e-06  6.880000e-06  7.000000e-06  
      2.050000e-05  2.120000e-05  2.180000e-05  3.600000e-05  3.710000e-05  
      6.500000e-05  6.750000e-05  1.010000e-04  1.050000e-04  1.160000e-04  
      1.180000e-04  1.880000e-04  1.920000e-04  2.250000e-03  3.740000e-03  
      1.700000e-02  2.000000e-02  5.000000e-02  2.000000e-01  2.700000e-01  
      3.300000e-01  4.700000e-01  6.000000e-01  7.500000e-01  8.610000e-01  
      1.200000e+00  1.500000e+00  1.850000e+00  3.000000e+00  4.300000e+00  
      6.430000e+00  2.000000e+01 
C61                 0.000000e+00  1.736482e-01  3.420201e-01  5.000000e-01  
      6.427876e-01  7.660444e-01  8.660254e-01  9.396926e-01  9.848078e-01  
      9.961947e-01  9.998477e-01  9.999619e-01  1.000000e+00 T
c
c
F31:N 115
FS31 46
C31    0.          0.17364818  0.34202014  0.5         0.64278761  0.76604444
       0.8660254   0.93969262  0.98480775  0.99756405  0.99862953  0.99939083  
       0.9998477   1.          T
c * added by ADVANTG
wwp:n 5.0 j 100 j -1 0 5.016581339e-09
