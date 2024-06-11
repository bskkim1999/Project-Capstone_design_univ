#include <stdio.h>

double bubbleSort(double *arr, int n) {
    double temp;
    double answer;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // swap arr[j] and arr[j+1]
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    return answer = arr[n / 2];
}

int main() {
    double arr[10] = { 3.1415, 2.7182, 1.6180, 0.5772, 1.4142, 2.2360, 3.1622, 1.7320, 0.6931, 2.6457 };
    double arr_2[10] = { 3.1415, 2.7182, 1.6180, 0.5772, 1.4142, 2.2360, 3.1622, 1.7320, 0.6931, 2.6457 };
    double arr_3[10] = { 3.1415, 2.7182, 1.6180, 0.5772, 1.4142, 2.2360, 3.1622, 1.7320, 0.6931, 2.6457 };

    int n = 10;
    double tmp=0;
    double tmp2 = 0;
    double tmp3 = 0;

    // Sort the array
    tmp = bubbleSort(arr, n);
    tmp2 = bubbleSort(arr_2, n);
    tmp3 = bubbleSort(arr_3, n);

    printf("Sorted array: ");

    for (int i = 0; i < 10; i++) {
        printf("%.4f    ", arr[i]);
    }
    printf("\n");

    for (int i = 0; i < 10; i++) {
        printf("%.4f    ", arr_2[i]);
    }

    printf("\n");

    for (int i = 0; i < 10; i++) {
        printf("%.4f    ", arr_3[i]);
    }

    printf("\n");

    printf("Median: %.4f\n", tmp);

    return 0;
}
