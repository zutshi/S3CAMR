'''
parses the file bbal.log
which took 4.8 days to compute
real    17391m21.106s
user    17260m31.784s
sys     125m51.456s

'''
import constraints as C
import fileops as fops

fr = fops.ReadLine('./bbal.log')

s = 'dummy'
x = C.IntervalCons((4.9,), (5.0,))


while s:
    if s.strip().find('1.0*x3 \\in ') == 0:
        ss = s.strip().replace('1.0*x3 \\in ', '')
        ic = C.IntervalCons(*map(lambda x: (float(x),), ss[1:-1].split(',')))
        if x & ic is not None:
            print ic
    s = fr.readline()

