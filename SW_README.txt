# Symbolic counting system 

Includes triangulation, reverse triangulation and calculation of vibrational waves. 

Utilizes consecutive -1, +1 bidirectional movement, strict odd/even definitions, dynamically scaling exponents, multiplication and division, as well as arithmetics.

## Symbol definitions

	'[' = Top left corner
	'}' = Bottom right corner
	']' = Top right corner
	'{' = Bottom left corner
	'(' and ')' = consecutive movement -1,+1 and maintains odd/even placement.
	'*' = multiply
	'/' = divide
	'0' = Data transmission placeholder(data transmission is unaffected by physical/wave properties)


Example 1(grid view,base principle):

			-10)-9	-2]-1  0   1[2   9(10
			-11(-12	-3(-4  0   4)3  12)11
			-14)-13	-6)-5  0   5(6  13(14
			-15{-16	-7(-8  0   8)7  16}15


Example 2(example 1 compiled to string format):

	'0' indicates data placeholder, '1[2' represents consecutive power, ')(' represents positive consecutive leap, '15}16' represents end of current positive leap "box". The exact same is true in reverse.

	-15{-16)(-2]-101[2)(15}16


Example 3(applying exponents to scale in tandem with ouroboros core calculations)

	Positive value leap -> scaled to exact box proportion via agent and DSLchain in code exec.

	01[150,002)(450,004}600,005 -> 256 squared scaled up or 65536 squared scaled down


Example 4(introducing multiplication/division while maintaining format structure):

	Maintaining consecutive -+1 structure allows for simplified and parameter removed calculation.

	01[52)(154}205(256*256)65537[265,538)(665,540}865,541

	Dynamically changes the size of the grid as well as any exponent parameter freely -> in tandem with ouroboros core calculation promotes near automatic calculation.

## Agent and Human comprehension section

	Designed to facilitate lightning fast agent computation and serve as the basis for converting raw numerical strings into vibrational wave data(sound bytes) for easier handling and validation.

	{-8)-7(-6]-5{-4)-3(-2]-1 0 1[+2)+3(+4}+5[+6)+7(+8}


	Basic string visualization of multibox counting, with the boxes being able to dynamically scale via agent code exec. tool and the utilization our ouroboros core calculations that are derived from real-world observation and data sets.
