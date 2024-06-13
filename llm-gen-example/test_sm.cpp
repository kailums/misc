#include <hip/hip_runtime.h>

int get_device_attribute(
    int attribute,
    int device_id)
{
    int device, value;
    if (device_id < 0) {
        hipGetDevice(&device);
    }
    else {
        device = device_id;
    }

    hipDeviceGetAttribute(&value, static_cast<hipDeviceAttribute_t>(attribute), device);
    return value;
}

int main() {
    int device_id = 0;
    int attribute = 97;
    auto attr_id0 = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto attr_id1 = hipDeviceAttributeSharedMemPerBlockOptin;
    int value = get_device_attribute(attribute, device_id);
    printf("value: %d\n", value);
    value = get_device_attribute(attr_id0, device_id);
    printf("value: %d, attr: %d\n", value, attr_id0);
    value = get_device_attribute(attr_id1, device_id);
    printf("value: %d, attr: %d\n", value, attr_id1);
    return 0;
}

