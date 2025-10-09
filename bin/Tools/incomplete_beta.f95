subroutine betai(a, b, x, betai_out)
  real, intent(in) :: a, b, x
  real, intent(out) :: betai_out
  real :: bt
  real :: betacf_out

  if (x < 0. .or. x > 1.) then
     print *, 'bad argument x in betai'
  end if

  if (x == 0. .or. x == 1.) then
     bt = 0.
  else
     bt = exp(log_gamma(a+b) - log_gamma(a) - log_gamma(b) + a*log(x) + b*log(1.-x))
  end if

  if (x < (a+1.)/(a+b+2.)) then
     call betacf(a, b, x, betacf_out)
     betai_out = bt*betacf_out/a
  else
     call betacf(b, a, 1.-x, betacf_out)
     betai_out = 1. - bt*betacf_out/b
  end if
end subroutine betai


subroutine betacf(a, b, x, betacf_out)
  real, intent(in) :: a, b, x
  real, intent(out) :: betacf_out
  integer, parameter :: MAXIT = 100
  real, parameter :: EPS = 3.e-7, FPMIN = 1.e-30
  integer :: m, m2
  real :: aa, c, d, del, h, qab, qam, qap

  qab = a+b
  qap = a+1.
  qam = a-1.
  c = 1.
  d = 1. - qab*x/qap
  if (abs(d) < FPMIN) d = FPMIN
  d = 1./d
  h = d

  do m = 1, MAXIT
     m2 = 2*m
     aa = m*(b-m)*x/((qam+m2)*(a+m2))
     d = 1. + aa*d
     if (abs(d) < FPMIN) d = FPMIN
     c = 1. + aa/c
     if (abs(c) < FPMIN) c = FPMIN
     d = 1./d
     del = d*c
     h = h*del
     if (abs(del-1.) < EPS) exit
  end do

  betacf_out = h
end subroutine betacf

