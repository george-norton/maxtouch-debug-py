#!/usr/bin/python3
import sys
import hid
import ctypes
import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
import argparse

"""
 Support parsing a few useful structures which we read/write.
"""
class InformationBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('family_id', ctypes.c_ubyte, 8),
        ('variant_id', ctypes.c_ubyte, 8),
        ('version', ctypes.c_ubyte, 8),
        ('build', ctypes.c_ubyte, 8),
        ('matrix_x_size', ctypes.c_ubyte, 8),
        ('matrix_y_size', ctypes.c_ubyte, 8),
        ('num_objects', ctypes.c_ubyte, 8),
    ]

class ObjectTableElement(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('type', ctypes.c_ubyte, 8),
        ('position_ls_byte', ctypes.c_ubyte, 8),
        ('position_ms_byte', ctypes.c_ubyte, 8),
        ('size_minus_one', ctypes.c_ubyte, 8),
        ('instances_minus_one', ctypes.c_ubyte, 8),
        ('report_ids_per_instance', ctypes.c_ubyte, 8)
    ]

class T6CommandProcessor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('reset', ctypes.c_ubyte, 8),
        ('backupnv', ctypes.c_ubyte, 8),
        ('calibrate', ctypes.c_ubyte, 8),
        ('reportall', ctypes.c_ubyte, 8),
        ('debugctrl', ctypes.c_ubyte, 8), # TODO, bitfield
        ('diagnostic', ctypes.c_ubyte, 8),
        ('debugctrl2', ctypes.c_ubyte, 8) # TODO, bitfield
    ]

# TODO! Set sensible VID/PID.
vendor_id     = 0xFEED
product_id    = 0x0000

usage_page    = 0xFF60
usage         = 0x61
report_length = 32

"""
Connect to the first candidate device we find.
"""
def get_raw_hid_interface():
    device_interfaces = hid.enumerate(vendor_id, product_id)
    raw_hid_interfaces = [i for i in device_interfaces if i['usage_page'] == usage_page and i['usage'] == usage]

    if len(raw_hid_interfaces) == 0:
        return None

    interface = hid.Device(path=raw_hid_interfaces[0]['path'])

    print(f"Manufacturer: {interface.manufacturer}")
    print(f"Product: {interface.product}")

    return interface

"""
Sanity check this device really is running the maxtouch debug firmware.
"""
def check_version():
    data = [
            0x00, # ReportID
            0x00, # Check version CMD
            0x9A, # Magic
            0x4D, # Magic
            0x00, # Version
            0x01  # Version
        ]
    data += [0x00] * (report_length + 1 - len(data)) # First byte is Report ID
    interface.write(bytes(data))
    response = interface.read(report_length, timeout=1000)
    if (response[0] != 0):
        raise Exception("Device reported error code "+str(response[0])) 

"""
The debug interface proxies I2C operations over the rawhid interface. This
function writes arbitary data to the maxtouch sensor at the supplied address.

The rawhid packets are 32 bytes, so the operation will be fragmented.

:param address: The I2C address to write to
:param data: The bytes to write
"""
def write_data(address, data):
    length = len(data)
    for offset in range(0, len(data), report_length - 4):
        write_length = min(length, report_length - 4)
        request = bytes([0x3, (address + offset) & 0xff, (address + offset) >> 8, write_length]) + data[offset:write_length+offset]
        if len(request) < report_length:
            request += bytes([0x00]) * (report_length - len(request))
        interface.write(request)
        response = interface.read(report_length, timeout=1000)
        length -= write_length
        if (response[0] != 0):
            raise Exception("Device reported error code "+str(response[0]))

"""
The debug interface proxies I2C operations over the rawhid interface. This
function reads data from the maxtouch sensor, from the supplied address.

The rawhid packets are 32 bytes, so the operation will be fragmented.

:param address: The I2C address to read from
:param length: The number of bytes to read
:retrun: The bytes read from the device
"""
def read_data(address, length):
    result = []
    for offset in range(0, length, report_length - 4):
        read_length = min(length, report_length - 4)
        request = bytes([0x2, (address + offset) & 0xff, (address + offset) >> 8, read_length] + [0x00] * (report_length - 4))
        interface.write(request)
        response = interface.read(report_length, timeout=1000)
        if (response[0] != 0):
            raise Exception("Device reported error code "+str(response[0]))

        result += response[4:4 + read_length]
        length -= read_length
    return bytes(result)

"""
Convienience function. The address map is not static, instead it is read
from an object table on the device. The object table includes the size and
location of each object in the address map. For now we only write to the first
instance of each object.

:param id: The id of the object to write
:param data: The bytes to write to the object
"""
def write_object(id, data):
    write_data(object_table[id]["address"], bytes(data))

"""
Convienience function. The address map is not static, instead it is read
from an object table on the device. The object table includes the size and
location of each object in the address map. For now we only read to the first
instance of each object.

:param id: The id of the object to read
:return: The bytes read from the object
"""
def read_object(id):
    return read_data(object_table[id]["address"], object_table[id]["size"])

"""
Convienience function for printing a cstruct once it has been populated.
"""
def print_object(obj):
    for field in obj._fields_:
        print(field[0], getattr(obj, field[0]))

"""
mathplotlib callback for populating a new frame.

:param frame: frame data
"""
cmap_range = [None, None]

def animate(frame):
    try:
        # Set the debug mode via the "diagnostic" register in the command object. This sets the page to 0.
        cmd = T6CommandProcessor.from_buffer_copy(bytes([0x00] * ctypes.sizeof(T6CommandProcessor)))
        if args.ref:
            cmd.diagnostic = 0x11 # Mutual Capacitance Reference Mode
        else:
            cmd.diagnostic = 0x10 # Mutual Capacitance Delta Mode

        write_object(6, cmd)
        # Switch the command to the next page command, but dont sent it yet. We will keep sending this
        # as we read more data.
        cmd.diagnostic = 0x1 # Next page

        sensor_data = bytearray()
        pages = math.ceil((sensor_nodes * 2) / 128)
        for page in range(pages):
            # The debug object is 130 bytes, it contains the debug mode, page, then 128 bytes of data
            debug_object = read_object(37)
            # TODO: improve this! If we read back the wrong page - try again.
            if debug_object[0] != 37 and debug_object[1] != page:
                time.sleep(0.1)
                debug_object = read_object(37)
                if debug_object[0] != 37 and debug_object[1] != page:
                    print("Got unexpected page..")
            sensor_data += debug_object[2:]
            # After we have read 128 bytes, we need to advance to the next page.
            if page < (pages-1):
                write_object(6, cmd)

        array = np.array(unpack("<"+"h"*(len(sensor_data)//2), sensor_data), dtype=np.short)[:sensor_nodes].reshape([information_block.matrix_x_size, information_block.matrix_y_size])
        image.set_array(array)
        if args.auto_clim:
            global cmap_range
            new_range = cmap_range.copy()
            if new_range[0] is None:
                new_range[0] = array.min()
            else:
                new_range[0] = min(new_range[0], array.min())
            if new_range[1] is None:
                new_range[1] = array.max()
            else:
                new_range[1] = min(new_range[1], array.max())
            if new_range != cmap_range:
                cmap_range = new_range
                print(f"Heatmap range {cmap_range[0]}..{cmap_range[1]}")
                image.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])
    except Exception as e:
        sys.exit(1)
    return image,

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", help="Show reference signal.", action="store_true")
    parser.add_argument("-c", "--clim", help="Heatmap colour limit (default: -128..127, or 0..32000 for reference mode).", type=int, nargs=2)
    parser.add_argument("-a", "--auto-clim", help="Auto Heatmap colour limit based on the min/max values in the image.", action="store_true")
    parser.add_argument("-R", "--recalibrate", help="Force recalibration before starting.", action="store_true")
    parser.add_argument("--cmap", help="Specify a matplotlib colour map for the heatmap (default: RdBu, or Blues for reference mode).p")
    args = parser.parse_args()

    """
    Find a device, for now this is not configurable.
    """
    with get_raw_hid_interface() as interface:
        check_version()

        """
        Read the sensor information block, and object table. This is at address 0 and it allows us to
        find all the rest of the objects supported by the sensor.
        """
        information_block = InformationBlock.from_buffer_copy(read_data(0, ctypes.sizeof(InformationBlock)))
        object_table = {}
        print(f"Sensor size is {information_block.matrix_x_size}x{information_block.matrix_y_size}")
        for index in range(information_block.num_objects):
            address = ctypes.sizeof(InformationBlock) + ctypes.sizeof(ObjectTableElement) * index
            object_table_entry = ObjectTableElement.from_buffer_copy(read_data(address, ctypes.sizeof(ObjectTableElement)))

            object_table[object_table_entry.type] = {
                "address": object_table_entry.position_ms_byte << 8 | object_table_entry.position_ls_byte,
                "instances": object_table_entry.instances_minus_one + 1,
                "size": object_table_entry.size_minus_one + 1
            }

        """
        If the user asked for a manual recalibration - do it here.
        """
        if args.recalibrate:
            # Send a calibrate command.
            cmd = T6CommandProcessor.from_buffer_copy(bytes([0x00] * ctypes.sizeof(T6CommandProcessor)))
            cmd.calibrate = 1
            write_object(6, cmd)

        """
        Build the matplotlib UI.
        """
        sensor_nodes = information_block.matrix_x_size * information_block.matrix_y_size
        array = np.zeros(shape=(information_block.matrix_x_size, information_block.matrix_y_size))
        fig = plt.figure()
        cmap = args.cmap if args.cmap else "Blues" if args.ref else "RdBu"
        image = plt.imshow(array, interpolation='none', animated=True, cmap=cmap)
        if args.clim:
            image.set_clim(vmin=args.clim[0], vmax=args.clim[1])
        else:
            if args.ref:
                image.set_clim(vmin=0, vmax=32000)
            else:
                image.set_clim(vmin=-128, vmax=127)
        anim = animation.FuncAnimation(
                fig,
                animate,
                interval = 0, # in ms - run as fast as we can
                blit = True,
                cache_frame_data = False
                )
        plt.show()
