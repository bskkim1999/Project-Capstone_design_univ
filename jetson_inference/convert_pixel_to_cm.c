#include <stdio.h>
#include <string.h>
#include <stdlib.h>

double convert_pixel_to_cm(double cm_, double pixel_, double object_pixel);


int main(void) {

    double tmp = convert_pixel_to_cm(5.0, 140.0, 30);

    printf("answer : %f", tmp);


    return 0;
}



double convert_pixel_to_cm(double cm_, double pixel_, double object_pixel) {
    /* depth에 따라서 cm당 픽셀크기가 달라지므로 고려해주어야 한다.
    //우선, 실험적으로 depth를 하나 정하여 Wpx를 구한다. 그런 다음,
    //Wpx를 cm로 변환한다.
    w : Wpx = 1cm : x(px)
    x(px) = Wpx / w      */

    double pixel_per_cm = pixel_ / cm_;   //1cm 당 몇 pixel인지 나옴

    return object_pixel / pixel_per_cm;
}
