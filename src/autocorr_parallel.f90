

subroutine autocov_calc_z(R_z,fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step)
    use omp_lib
    ! dummy args declarations
    implicit none

    integer(4),intent(in) :: NCL1, NCL2, NCL3, max_z_step
    real(8), dimension(max_z_step,NCL2,NCL1),intent(out) :: R_z
    real(8), dimension(NCL3,NCL2,NCL1),intent(in) :: fluct1,fluct2
    
    !main body
    integer(4) :: i,j

    R_z=0
#ifdef COMP
    !$OMP PARALLEL DO &
    !$OMP REDUCTION(+:R_z)&
    !$OMP SCHEDULE(DYNAMIC)
#else
    !$OMP PARALLEL DO
    !$OMP& REDUCTION(+:R_z)
    !$OMP& SCHEDULE(DYNAMIC)
#endif
    do i = 1,max_z_step
        do j = 1, NCL3-max_z_step
            R_z(i,:,:) = R_z(i,:,:) + fluct1(j,:,:)*fluct2(i+j,:,:)
        enddo
    enddo

    !$OMP END PARALLEL DO

    
end subroutine
subroutine autocov_calc_x(R_x,fluct1,fluct2,NCL3,NCL2,NCL1,max_x_step)
    use omp_lib
    implicit none

    integer(4), intent(in) :: NCL1, NCL2, NCL3, max_x_step
    real(8), dimension(max_x_step,NCL2,NCL1-max_x_step),intent(out) :: R_x
    real(8), dimension(NCL3,NCL2,NCL1),intent(in) :: fluct1,fluct2

    !main body
    integer(4) :: i,j,k
    R_x=0

#ifdef COMP
    !$OMP PARALLEL DO &
    !$OMP REDUCTION(+:R_x)&
    !$OMP SCHEDULE(DYNAMIC)
#else
    !$OMP PARALLEL DO
    !$OMP& REDUCTION(+:R_x)
    !$OMP& SCHEDULE(DYNAMIC)
#endif
    do k = 1,NCL3
        do i = 1, max_x_step
            do j=1,NCL1-max_x_step
                R_x(j,:,i) = R_x(j,:,i) + fluct1(j,:,k)*fluct2(j+i,:,k)
            enddo
        enddo
    enddo

    !$OMP END PARALLEL DO




end subroutine
