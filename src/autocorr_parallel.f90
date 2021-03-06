

subroutine autocov_calc_z(fluct1,fluct2,R_z,max_z_step,NCL1,NCL2,NCL3)
    use omp_lib
    ! dummy args declarations
    implicit none

    integer(4),intent(in) :: NCL1, NCL2, NCL3, max_z_step
    real(8), dimension(max_z_step,NCL2,NCL1),intent(inout) :: R_z
    real(8), dimension(NCL3,NCL2,NCL1),intent(in) :: fluct1,fluct2
    
!f2py intent(in,out) :: R_z
    
    !main body
    integer(4) :: i,j

    R_z=0

    !$OMP PARALLEL DO &
    !$OMP SCHEDULE(DYNAMIC)&
    !$OMP COLLAPSE(2)
    do i = 1,max_z_step
        do j = 1, NCL3-max_z_step
            R_z(i,:,:) = R_z(i,:,:) + fluct1(j,:,:)*fluct2(i+j,:,:)
        enddo
    enddo

    !$OMP END PARALLEL DO

    
end subroutine
subroutine autocov_calc_x(fluct1,fluct2,R_x,max_x_step,NCL3,NCL2,NCL1)
    use omp_lib
    implicit none

    integer(4), intent(in) :: NCL1, NCL2, NCL3, max_x_step
    real(8), dimension(max_x_step,NCL2,NCL1-max_x_step),intent(inout) :: R_x
    real(8), dimension(NCL3,NCL2,NCL1),intent(in) :: fluct1,fluct2

!f2py intent(in,out) :: R_x

    !main body
    integer(4) :: i,j,k
    R_x=0


    !$OMP PARALLEL DO &
    !$OMP SCHEDULE(DYNAMIC)&
    !$OMP COLLAPSE(2)

    
    do i = 1, max_x_step
        do j=1,NCL1-max_x_step
            do k = 1,NCL3
                R_x(j,:,i) = R_x(j,:,i) + fluct1(k,:,j)*fluct2(k,:,j+i)/NCL3
            enddo
        enddo
    enddo

    !$OMP END PARALLEL DO




end subroutine
