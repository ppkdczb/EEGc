# encoding: utf-8

'''

Usage
-----

The ``demo_...()`` functions in ``pygds.py`` are an example of how to use ``pygds.GDS``.

Basic usage:

.. code:: py

    >>> import pygds
    >>> d = pygds.GDS()

``pygds.GDS`` hides the API differences between ``g.USBamp``, ``g.HIamp`` and ``g.Nautilus``.
E.g. ``d.GetImpedanceEx()`` calls the right function of::

    GDS_GUSBAMP_GetImpedanceEx()
    GDS_GHIAMP_GetImpedanceEx()
    GDS_GNAUTILUS_GetImpedanceEx()

Similarly the configuration names are unified.
For example, Trigger means TriggerEnabled, TriggerLinesEnabled or DigitalIOs. See ``name_maps``.
The device-specific names also work:

.. code:: py

    >>> d.TriggerEnabled == d.Trigger
    True

For one device, the configuration fields are members of the device object:

.. code:: py

    >>> d.Trigger = True
    >>> d.SetConfiguration()

For more devices, use the ``Configs`` list:

.. code:: py

    >>> for c in d.Configs:
    ...     c.Trigger = True
    >>> d.SetConfiguration()

``pygds.configure_demo()`` configures all available channels:

.. code:: py

    >>> pygds.configure_demo(d,testsignal=1)
    >>> d.SetConfiguration()

To acquire a fixed number of samples, please use:

.. code:: py

    >>> a = d.GetData(d.SamplingRate)
    >>> a.shape[0] == d.SamplingRate
    True

To acquire a dynamic number of samples, provide a function ``more(samples)``.

A ``pygds.Scope`` object can be used as ``more`` parameter of ``GetData()``.
When closing the scope Window acquisition stops.

.. code:: py

    >>> scope = pygds.Scope(1/d.SamplingRate, title="Channels: %s", ylabel = u"U[μV]")
    >>> a = d.GetData(d.SamplingRate//2,scope)
    >>> del scope
    >>> a.shape[1]>=d.N_electrodes
    True

Don't forget ``del scope`` before repeating this.

To remove a GDS object manually, do:

.. code:: py

    >>> d.Close()
    >>> del d

In the doctest samples, this is done to make the next test succeed.
For a session where only one GDS object is used, there is no need to do this.

'''


from __future__ import nested_scopes
from __future__ import generators
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import print_function

import sys
import re
import os
import os.path
import cffi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import atexit
import socket

# maximum acquisition timeout in seconds
AcquisitionTimeout = 10

def _gds_running():
    try:
        import win32serviceutil  # part of pywin32
        gds_state = win32serviceutil.QueryServiceStatus('GDS')
        return gds_state[1] == 4  # running
    except:
        return False


try:
    from queue import Queue
except:
    input = raw_input
    from Queue import Queue


'''

Reference
---------

pygds global
~~~~~~~~~~~~

'''

from threading import Thread
from timeit import default_timer as timer

'''
``pygds`` needs these header files and the DLL.

If they are in a different location,
you must call ``pygds.Initialize()`` manually and provide the right paths.

'''
gNEEDaccessHeaders = [
    os.path.expandvars(r"%USERPROFILE%/Documents/gtec/gNEEDaccessClientAPI/C/GDSClientAPI.h"),
    os.path.expandvars(r"%USERPROFILE%/Documents/gtec/gNEEDaccessClientAPI/C/GDSClientAPI_gHIamp.h"),
    os.path.expandvars(r"%USERPROFILE%/Documents/gtec/gNEEDaccessClientAPI/C/GDSClientAPI_gNautilus.h"),
    os.path.expandvars(r"%USERPROFILE%/Documents/gtec/gNEEDaccessClientAPI/C/GDSClientAPI_gUSBamp.h")]

#gds_dll_client = os.path.expandvars(r"%USERPROFILE%/Documents/gtec/gNEEDaccessClientAPI/C/x64/GDSClientAPI.dll")
#gds_dll_client = r"C:/Program Files/gtec/gNEEDaccess/GDSClientAPI.dll"
#gds_dll_standalone = r"C:/Program Files/gtec/gNEEDaccess/GDSServer.dll"
gds_dll_client = r"GDSClientAPI.dll"
gds_dll_standalone = r"GDSServer.dll"


class GDSError(Exception):
    '''
    This is the exception that is raised in case of a g.NEEDaccess API error.

    '''

    def __init__(self, message=None):
        Exception.__init__(self, message)


def _chk(gds_res):
    if gds_res and gds_res.ErrorCode != ERROR_SUCCESS:
        raise GDSError(_ffi.string(gds_res.ErrorMessage).decode())


_api = []
_defs = {}
_ndefs = {}  # numerical defs
_enums = []

'''
``pygds.OpenDevices`` contains all objects of ``pygds.GDS()``.
It is used to clean up when exiting Python.

'''
OpenDevices = None

g_gds_dll = ""


def Initialize(
    gds_headers=gNEEDaccessHeaders,  # default header files used
    gds_dll=None
):
    '''
    Initializes ``pygds``. This is done automatically at ``import pygds``.

    If the GDS service is running, then ``GDSClientAPI.dll`` is used (which supports remote acquisition); otherwise, ``GDSServer.dll`` (local machine only).
    To manually change, call ``Uninitialize()`` first, followed by a call to ``Initialize(gds_dll=...)``.
    Set the ``gds_dll`` parameter to the appropriate library using either ``gds_dll_client`` or ``gds_dll_standalone`` symbol.
    For example, calling ``Initialize(gds_dll=pygds.gds_dll_standalone)`` is equivalent to calling ``Initialize(gds_dll="GDSServer.dll")``.

    ``pygds.Initialize()``:

    - populates the ``pygds`` namespace with definitions from the GDS headers. The ``GDS_`` prefix is dropped
    - loads the selected GDS client DLL
    - calls ``GDS_Initialize()``

    If g.NEEDaccess is installed in a non-standard location, then ``pygds.Initialize()`` will fail.
    Then you need to call ``pygds.Initialize()`` manually and
    provide the header file paths and the DLL path as parameters.

    The return value is True if initialization succeeded.

    '''

    global OpenDevices
    if OpenDevices != None:
        return

    if gds_dll == None:
        if _gds_running():
            gds_dll = gds_dll_client
        else:
            gds_dll = gds_dll_standalone

    global g_gds_dll
    g_gds_dll = gds_dll
    #print("using %s"%gds_dll)

    def _cdefines(lns):
        d = dict()
        for line in lns:
            match = re.search(
                r'#define\s+([A-Z]\w*)\s+([0-9A-Za-z."\-* ]+)', line)
            if match:
                d[match.group(1)] = str(eval(match.group(2)))
        return d

    def _ccode(lns, defs):
        ret = []
        for a in lns:
            if (not re.search(r'^\s*#', a) and
                not re.search(r'extern', a) and
                not re.search(r'^\s*//', a) and
                not re.search(r'^\s*}\s*$', a) and
                    not re.search(r'^extern', a)):
                for dfn, val in defs.items():
                    a = a.replace(dfn, val)
                ret += [a]
        return ret

    def _defs_api(headerfile, defs, api, enums):
        lns = []
        with open(headerfile) as f:
            lns = f.readlines()
        thisdef = _cdefines(lns)
        thisdef['GDSCLIENTAPI_API'] = ''
        defs.update(thisdef)
        thisapi = _ccode(lns, defs)
        api.extend(thisapi)
        for i in range(len(lns)):
            if 'typedef enum' in lns[i]:
                for k in range(len(lns)-i):
                    ln = lns[i+k]
                    mo = re.match(r"\s*}\s*(\w+)\s*;", ln)
                    if mo:
                        enums.append(mo.group(1))

    try:
        _api.clear()
        _enums.clear()
        _defs.clear()
        _ndefs.clear()
    except:
        pass

    tapi = []
    for headerfile in gds_headers:
        try:
            os.stat(headerfile)  # will raise exception if file does not exist
        except:
            return False
        _defs_api(headerfile, _defs, tapi, _enums)

    #tapi.extend(_ccode(["GDS_RESULT GDS_GUSBAMP_GetTemperature(GDS_HANDLE connectionHandle, char (*deviceName)[DEVICE_NAME_LENGTH_MAX], float *temperature);",
    #                    "GDS_RESULT GDS_GHIAMP_GetTemperature(GDS_HANDLE connectionHandle, char (*deviceName)[DEVICE_NAME_LENGTH_MAX], float *temperature);"], _defs))

    for a in tapi:
        mo = re.match(
            r"^(\W*)(?P<type>\w+)\s*,\s*\*(?P<pointer>P(?P=type));", a)
        if mo:
            _api.append(
                re.sub(r"(?P<type>\w+)\s*,\s*\*(?P<pointer>P(?P=type));", r"\1;", a))
            _api.append("typedef " + mo.group('type') +
                        ' *' + mo.group('pointer') + ";")
        else:
            _api.append(a)

    for k, x in _defs.items():
        try:
            _ndefs[k] = int(x)
        except ValueError:
            _ndefs[k] = x

    def strip_gds(d): return {k.replace('GDS_', ''): v for k, v in d.items()}

    globals().update(strip_gds(_ndefs))

    _gdshdrs = '\n'.join(_api)

    global _ffi
    try:
        del globals()['_ffi']
    except:
        pass
    _ffi = cffi.FFI()
    try:
        _ffi.cdef(_gdshdrs, override=True)
    except:
        pass

    global _ffi_dll
    try:
        del globals()['_ffi_dll']
    except:
        pass
    try:
        _ffi_dll = _ffi.dlopen(gds_dll)
    except:
        return False

    _chk(_ffi_dll.GDS_Initialize())
    atexit.register(Uninitialize)

    for enum in _enums:
        enumtype = _ffi.typeof(enum)
        globals().update({enum.replace('GDS_', ''): enumtype})
        globals().update(strip_gds(enumtype.relements))

    globals()['NULL'] = _ffi.NULL
    OpenDevices = {}
    return True


def Uninitialize():
    '''
    Clean up is done automatically when exiting Python.

    ``Uninitialize()`` tries not to block, by taking into account these GDS API behaviors:

    - ``GDS_Uninitialize()`` blocks, if called after calling ``GDS_Disconnect()`` on all connections.
    - On the other hand, to prevent a freeze, one must call ``GDS_Uninitialize()``, if no device was ever connected, but ``GDS_Initialize()`` had been called.

    '''

    global OpenDevices
    lenserials = 0
    if OpenDevices:
        serials = list(OpenDevices.keys())
        lenserials = len(serials)
        for s in serials:
            del OpenDevices[s]
    OpenDevices = None
    if lenserials == 0:
        _ffi_dll.GDS_Uninitialize()


if __name__ != '__main__':
    if not Initialize():
        print("pygds.Initialize failed to load either header or DLL. Call it manually and provide the right paths.")

SERVER_IP = "127.0.0.1"
SERVER_PORT = 50223
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 50224


def _encoded(bytes_or_str):
    try:
        return bytes_or_str.encode()
    except:
        return bytes_or_str


def _this_ip(server_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((server_ip, 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def _server_client(server_ip):
    serverip = _encoded(server_ip)
    if not serverip.startswith(b'127.'):
        clientip = _encoded(_this_ip(server_ip))
    else:
        clientip = _encoded(CLIENT_IP)
    return serverip, clientip


class ConnectedDevices(list):
    '''
    Lists all connected devices in a list of type ``[(serial, devicetype, inuse)]``:

    .. code:: py

        >>> import pygds
        >>> cd = pygds.ConnectedDevices()

    This is used by the ``pygds.GDS`` constructor.
    Use it separately only if you don't want to instantiate a pygds.GDS object,
    but still want to find out which devices are connected.

    '''

    def __init__(self,
                 server_ip=SERVER_IP
                 ):

        list.__init__(self)
        self._cd = _ffi.new('GDS_DEVICE_CONNECTION_INFO **')
        self._cdc = _ffi.new('size_t *')
        self.serverip, self.clientip = _server_client(server_ip)
        _chk(_ffi_dll.GDS_GetConnectedDevices((self.serverip, SERVER_PORT),
                                              (self.clientip, CLIENT_PORT), self._cd, self._cdc))
        for i in range(self._cdc[0]):
            inuse = bool(self._cd[0][i].InUse)
            for j in range(self._cd[0][i].ConnectedDevicesLength):
                en = self._cd[0][i].ConnectedDevices[j]
                self.append(
                    (_ffi.string(en.Name).decode(), en.DeviceType, inuse))

    def __del__(self):
        _chk(_ffi_dll.GDS_FreeConnectedDevicesList(self._cd, self._cdc[0]))

    def find(self,
             wanted_type,  # a DEVICE_TYPE_XXX constant
             exclude_serials=None  # list of serials to exclude
             ):
        '''ConnectedDevices.

        Find a device that is currently not in use by type.

        .. code:: py

            >>> import pygds
            >>> cd = pygds.ConnectedDevices()
            >>> hiamp = cd.find(pygds.DEVICE_TYPE_GHIAMP)
            >>> hiamp is None or len(hiamp.split('.'))>0
            True

        '''

        for name, type, inuse in self:
            if not inuse:
                if type != DEVICE_TYPE_NOT_SUPPORTED:
                    if wanted_type == DEVICE_TYPE_NOT_SUPPORTED or type == wanted_type:
                        if exclude_serials == None or name not in exclude_serials:
                            return name


'''
``name_maps`` provides common names for the device-specific configuration fields
in order to facilitate code reuse across devices.

'''
name_maps = {
        'GDS_GUSBAMP_CONFIGURATION':
            {
                "SamplingRate": "SampleRate",
                "Counter": "CounterEnabled",
                "Trigger": "TriggerEnabled",
                "DI": "TriggerEnabled",
            },
        'GDS_GHIAMP_CONFIGURATION':
            {
                "SampleRate": "SamplingRate",
                "Counter": "CounterEnabled",
                "TriggerEnabled": "TriggerLinesEnabled",
                "Trigger": "TriggerLinesEnabled",
                "DI": "TriggerLinesEnabled",
            },
        'GDS_GNAUTILUS_CONFIGURATION':
            {
                "SampleRate": "SamplingRate",
                "Trigger": "DigitalIOs",
                "TriggerEnabled": "DigitalIOs",
                "DI": "DigitalIOs",
            },
        'GDS_GUSBAMP_CHANNEL_CONFIGURATION':
            {
                "Enabled": "Acquire",
                "ReferenceChannel": "BipolarChannel",
            },
        'GDS_GHIAMP_CHANNEL_CONFIGURATION':
            {
                "Enabled": "Acquire",
                "BipolarChannel": "ReferenceChannel",
            },
        'GDS_GNAUTILUS_CHANNEL_CONFIGURATION':
            {
                "Acquire": "Enabled",
                "ReferenceChannel": "BipolarChannel",
            },
        'GDS_GUSBAMP_SCALING':
            {
                "Factor": "ScalingFactor",
            },
}


class _ffi_struct_member:
    def __init__(self, actualname):
        self.actualname = actualname

    def __get__(self, aninstance):
        res = getattr(aninstance._ffi_struct, self.actualname)
        try:
            tp = _ffi.typeof(res)
            if tp.kind == 'struct':
                return _ffi_struct_wrap(res, name_maps.get(tp.cname, {}))
            elif tp.kind == 'array':
                mp = name_maps.get(tp.cname.split('[')[0], {})
                try:
                    return [_ffi_struct_wrap(r, mp) for r in res]
                except:
                    return [r for r in res]
        except:
            return res

    def __set__(self, aninstance, value):
        return setattr(aninstance._ffi_struct, self.actualname, value)

    def _to_python(self, aninstance):
        res = getattr(aninstance._ffi_struct, self.actualname)
        try:
            tp = _ffi.typeof(res)
            if tp.kind == 'struct':
                return _ffi_struct_wrap(res, name_maps.get(tp.cname, {}))._to_python()
            elif tp.kind == 'array':
                mp = name_maps.get(tp.cname.split('[')[0], {})
                try:
                    return [_ffi_struct_wrap(r, mp)._to_python() for r in res]
                except:
                    return [r for r in res]
        except:
            pass
        return res


class _ffi_struct_wrap(object):
    def __init__(self, ffi_struct, name_map={}):
        self._ffi_struct = ffi_struct
        self._name_map = name_map
        tp = _ffi.typeof(self._ffi_struct)
        if tp.kind == 'pointer':
            tp = _ffi.typeof(self._ffi_struct[0])
        if tp.kind == 'struct':
            for k, v in tp.fields:
                if hasattr(self, k):
                    del self.__dict__[k]
                setattr(self, k, _ffi_struct_member(k))
            for k, v in name_map.items():
                if hasattr(self, k):
                    del self.__dict__[k]
                setattr(self, k, _ffi_struct_member(v))

    def __getattribute__(self, key):
        attr = super(_ffi_struct_wrap, self).__getattribute__(key)
        if isinstance(attr, _ffi_struct_member):
            try:
                return attr.__get__(self)
            except AttributeError:
                raise AttributeError(
                    "Property '{}' on object {} does not allow read access".format(key, self))
        else:
            return attr

    def __setattr__(self, key, value):
        try:
            attr = super(_ffi_struct_wrap, self).__getattribute__(key)
        except AttributeError:
            attr = None
        if isinstance(attr, _ffi_struct_member):
            try:
                attr.__set__(self, value)
            except AttributeError:
                raise AttributeError(
                    "Property '{}' on object {} does not allow write access".format(key, self))
        else:
            super(_ffi_struct_wrap, self).__setattr__(key, value)

    def __delattr__(self, key):
        attr = super(_ffi_struct_wrap, self).__getattribute__(key)
        if isinstance(attr, _ffi_struct_member):
            try:
                attr.__delete__(self)
            except AttributeError:
                raise AttributeError(
                    "Property '{}' on object {} does not allow deletion".format(key, self))
        super(_ffi_struct_wrap, self).__delattr__(key)

    def __str__(self):
        return str(self._to_python())

    def __repr__(self):
        return str(self)

    def _to_python(self):
        def __to_python(v):
            try:
                return v._to_python(self)
            except:
                return v
        return {k: __to_python(v) for k, v in self.__dict__.items() if not k.startswith('_') and k not in self._name_map}


class _config_wrap(_ffi_struct_wrap):
    def __init__(self, ffi_struct):
        self.DeviceType = ffi_struct.DeviceInfo.DeviceType
        cstr = ['GDS_GUSBAMP_CONFIGURATION',
                'GDS_GHIAMP_CONFIGURATION',
                'GDS_GNAUTILUS_CONFIGURATION'][self.DeviceType-1]
        _ffi_struct_wrap.__init__(self, _ffi.cast(
            cstr+' *', ffi_struct.Configuration)[0], name_maps[cstr])


'''

Scope
~~~~~

'''


class Scope:
    '''
    ``Scope`` makes a live update of a Matplotlib diagram and thus simulates an oscilloscope:

    .. code:: py

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pygds import Scope
        >>> import time
        >>> f = 10
        >>> scope=Scope(1/f)
        >>> t = np.linspace(0,100,100)/f
        >>> scope(np.array([np.sin(t+i/2) for i in range(10)]))
        True
        >>> time.sleep(0.1)
        >>> scope(np.array([np.sin(t+i/3) for i in range(10)]))
        True
        >>> time.sleep(0.1)
        >>> scope(np.array([np.sin(t+i/4) for i in range(10)]))
        True
        >>> time.sleep(0.1)
        >>> scope(np.array([np.sin(t+i/5) for i in range(10)]))
        True
        >>> del scope

    ``Scope`` can be used as the ``more`` argument of ``GetData()`` to have a live view on the data.

    To use ``Scope`` as a regular diagram, set ``modal=True``.

    The object's ``__call__(self,scan)`` displays the scans.
    On the first call to the object (via ``__call__()``), the diagram is initialized.
    The ``scans`` parameter of ``__call__()`` is an ``(n,ch)`` numpy array. It must have the same shape at every call.


    '''

    def __init__(self,
                 # to give the x axis time units. Use Use 1/SamplingRate to produce a second as unit.
                 time_factor,
                 modal=False,  # set this to True if not used as scope
                 # {channel index:subplot index}. It allows to distribute channels to subplots.
                 subplots=None,
                 # key words of plt.subplots. In case of more subplots, make more entries per key word, like ylabel=('V','I')
                 **subplot_kw
                 ):

        if not modal:
            plt.ion()
            plt.show()
        self.time_factor = time_factor
        self.n0 = 0
        self.ax = None
        self.lines = None
        self.fig = None
        self.subplot_kw = subplot_kw
        self.subplots = {0: 0} if not subplots else subplots

    def __del__(self):
        plt.ioff()

    def __call__(self, scans):
        nn = scans.shape[0]
        t = np.linspace(self.n0, self.n0+nn, nn)*self.time_factor
        self.n0 = self.n0 + nn
        mins = {}
        maxs = {}

        def redraw():
            for j, ax in enumerate(self.ax):
                ax.set_xlim(t[0], t[-1])
                xj, nj = maxs[j], mins[j]
                dlta = 0.05*(xj-nj)
                ax.set_ylim(nj-dlta, xj+dlta)
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except NotImplementedError:
                pass
        if isinstance(self.ax, type(None)):
            nrows = max(self.subplots.values())+1
            self.fig, self.ax = plt.subplots(
                nrows=nrows, ncols=1, sharex=True, subplot_kw=self.subplot_kw if nrows == 1 else None)
            if not isinstance(self.ax, np.ndarray):
                self.ax = (self.ax,)
            self.lines = []
            for i, s in enumerate(scans.T):
                l = Line2D(t, s, c=np.random.rand(3,))
                self.lines.append(l)
                j = self.subplots.get(i, 0)
                self.ax[j].add_line(l)
                sx, sn = s.max(), s.min()
                mins[j] = min(mins[j], sn) if j in mins else sn
                maxs[j] = max(maxs[j], sx) if j in maxs else sx
            if nrows > 1:
                for j, ax in enumerate(self.ax):
                    ax.set(**{k: v[j] for k, v in self.subplot_kw.items()})
            ttl = self.ax[0].get_title()
            if '%s' in ttl:
                self.ax[0].set_title(ttl % len(self.lines))
            redraw()
        else:
            if not plt.fignum_exists(self.fig.number):
                return False
            for i, s in enumerate(scans.T):
                self.lines[i].set_data(t, s)
                j = self.subplots.get(i, 0)
                sx, sn = s.max(), s.min()
                mins[j] = min(mins[j], sn) if j in mins else sn
                maxs[j] = max(maxs[j], sx) if j in maxs else sx
            redraw()
        return True


'''

GDS
~~~

'''

class GDS(_config_wrap):
    '''
    The ``pygds.GDS`` class initializes the connection to g.NEEDaccess server.

    The constructor:

    - initializes the connection to the wanted device(s) and
    - fetches the configuration(s).

    *gds_device*: can be:

        - omitted (default)
        - the first letter of the serial
        - one of DEVICE_TYPE_GUSBAMP, DEVICE_TYPE_GHIAMP or DEVICE_TYPE_GNAUTILUS
        - a single serial
        - comma-separated serials

    *exclude_serials*: a list or set of serials to ignore. Default: ``None``.

    *open_exclusively*: ``True`` to prevent a newly created data acquisition session from 
                        being opened or accessed by another GDS client instance with 
                        the same list of device serials; otherwise, ``False``. Default: ``True``.

        - To connect to an already existing data acquisition session and listen to its 
          data stream, both clients have to open the session non-exclusively (using ``False``)
          and with the same list of devices.

        - The first client that opens the same list of devices is the session creator.
          Only the session creator can configure the device and start/stop acquisition 
          on the device itself. Other clients are participants and can only retrieve data
          from a running data acquisition stream started by the session creator.

        - ``gds_device`` must specify the serial number(s) explicitly and cannot be omitted 
          if a participant wants to connect to an already existing data acquisition session.

    *server_ip*: the IP address of the GDS server. Default: pygds.SERVER_IP

        - The g.NEEDaccess server port is pygds.SERVER_PORT and it is fixed.

        - The client by default is pygds.CLIENT_IP and pygds.CLIENT_PORT.
          For a remote ``server_ip``, the local IP is automatically determined.

    Without parameters the localhost g.NEEDaccess server is used
    and the first available device is connected.

    For one device the configuration fields are members of the GDS object.
    For more devices, every configuration is an entry in the ``Configs`` member.

    g.USBamp config:

    +-----------------------------------+------------------------------------------------------------------------+
    | Name                              | String holding the serial number of g.USBamp (e.g. ``UB-2014.01.02``)  |
    +-----------------------------------+------------------------------------------------------------------------+
    | DeviceType                        | ``pygds.DEVICE_TYPE_XXX`` constant representing the device type        |
    |                                   | (predefined, must not be changed)                                      |
    +-----------------------------------+------------------------------------------------------------------------+
    | SamplingRate                      | Specify the sampling frequency of g.HIamp in Hz as unsigned integer    |
    +-----------------------------------+------------------------------------------------------------------------+
    | NumberOfScans                     | Specify the buffering block size as unsigned short, possible values    |
    |                                   | depend on sampling rate, use function ``GetSupportedSamplingRates()``  |
    |                                   | to get recommended values.                                             |
    +-----------------------------------+------------------------------------------------------------------------+
    | CommonGround                      | Array of 4 bool elements to enable or disable common ground            |
    +-----------------------------------+------------------------------------------------------------------------+
    | CommonReference                   | Array of 4 bool values to enable or disable common reference           |
    +-----------------------------------+------------------------------------------------------------------------+
    | ShortCutEnabled                   | Bool enabling or disabling g.USBamp shortcut                           |
    +-----------------------------------+------------------------------------------------------------------------+
    | CounterEnabled                    | Show a counter on first recorded channel that is incremented with      |
    |                                   | every block transmitted to the PC. Overruns at 1000000.                |
    +-----------------------------------+------------------------------------------------------------------------+
    | TriggerEnabled                    | Scan the digital trigger channel with the analog inputs                |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Enabled   | Apply internal test signal to all inputs                               |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.WaveShape | ``pygds.GUSBAMP_WAVESHAPE_XXX`` constant representing the  wave shape  |
    |                                   | of the internal test signal. Can be 0=square, 1=sawtooth, 2=sine,      |
    |                                   | 3=DRL, or 4=noise                                                      |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Amplitude | The amplitude of the test signal (can be -250 to 250 mV)               |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Offset    | The offset of the test signal (can be -200 to 200 mV)                  |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Frequency | The frequency of the test signal (can be 1 to 100 Hz)                  |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels                          | Array of g.USBamp channel configurations holding properties for each   |
    |                                   | analog channel                                                         |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].ChannelNumber         | Unsigned integer holding the channel number of the analog channel      |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].Acquire               | Bool value selecting the channel for data acquisition                  |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].BandpassFilterIndex   | Perform a digital bandpass filtering of the input channels. Use        |
    |                                   | ``GetBandpassFilters()`` to get filter indices.                        |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].NotchFilterIndex      | Perform a bandstop filtering to suppress the power line frequency of   |
    |                                   | 50 Hz or 60 Hz. Use ``GetNotchFilters()`` to get filter indices.       |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].BipolarChannel        | Select a channel number as reference channel for an analog channel     |
    +-----------------------------------+------------------------------------------------------------------------+



    g.HIamp config:

    +-----------------------------------+------------------------------------------------------------------------+
    | Name                              | String holding the serial number of g.HIamp (e.g. ``HA-2014.01.02``)   |
    +-----------------------------------+------------------------------------------------------------------------+
    | DeviceType                        | ``pygds.DEVICE_TYPE_XXX`` constant representing the device type        |
    |                                   | (predefined, must not be changed)                                      |
    +-----------------------------------+------------------------------------------------------------------------+
    | SamplingRate                      | Specify the sampling frequency of g.Hiamp in Hz as unsigned integer    |
    +-----------------------------------+------------------------------------------------------------------------+
    | NumberOfScans                     | Specify the buffering block size as unsigned short, possible values    |
    |                                   | depend on sampling rate, use function ``GetSupportedSamplingRates()``  |
    |                                   | to get recommended values                                              |
    +-----------------------------------+------------------------------------------------------------------------+
    | CounterEnabled                    | Show a counter on first recorded channel which is incremented with     |
    |                                   | every block transmitted to the PC. Overruns at 1,000,000               |
    +-----------------------------------+------------------------------------------------------------------------+
    | TriggerLinesEnabled               | Scan the digital trigger channel with the analog inputs                |
    +-----------------------------------+------------------------------------------------------------------------+
    | HoldEnabled                       | Enable signal hold                                                     |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels                          | Array of g.HIamp channel configurations holding                        |
    |                                   | properties for each analog channel                                     |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].ChannelNumber         | Unsigned integer holding the channel number of the analog channel      |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].Acquire               | Bool value selecting the channel for data acquisition                  |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].BandpassFilterIndex   | Perform a digital bandpass filtering of the input channels. Use        |
    |                                   | ``GetBandpassFilters()`` to get filter indices                         |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].NotchFilterIndex      | Perform a bandstop filtering to suppress the power line frequency of   |
    |                                   | 50 Hz or 60 Hz. Use ``GetNotchFilters()`` to get filter indices        |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].ReferenceChannel      | Select a channel number as reference channel for an analog channel     |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Enabled   | Apply internal test signal to all inputs (requires shortcut of all     |
    |                                   | analog channels to ground)                                             |
    +-----------------------------------+------------------------------------------------------------------------+
    | InternalSignalGenerator.Frequency | Specify the frequency of the test signal.                              |
    |                                   | Fix: Amplitude = -Offset = 7.62283 mV.                                 |
    +-----------------------------------+------------------------------------------------------------------------+

    g.Nautilus config:

    +-----------------------------------+------------------------------------------------------------------------+
    | Name                              | String holding the serial number of g.Nautilus (e.g. NA-2014.07.67)    |
    +-----------------------------------+------------------------------------------------------------------------+
    | DeviceType                        | ``pygds.DEVICE_TYPE_XXX`` constant representing the device type        |
    +-----------------------------------+------------------------------------------------------------------------+
    | SamplingRate                      | Specify the sampling frequency of g.Nautilus in Hz as unsigned integer |
    +-----------------------------------+------------------------------------------------------------------------+
    | NumberOfScans                     | Specify the buffering block size as unsigned short, possible values    |
    |                                   | depend on sampling rate, use function ``GetSupportedSamplingRates()``  |
    |                                   | to get recommended values                                              |
    +-----------------------------------+------------------------------------------------------------------------+
    | InputSignal                       | ``pygds.GNAUTILUS_INPUT_SIGNAL_XXX`` constant representing the type of |
    |                                   | input signal. Can be 0=electrode, 1=shortcut, or 5=test signal.        |
    +-----------------------------------+------------------------------------------------------------------------+
    | NoiseReduction                    | Bool value enabling noise reduction for g.Nautilus                     |
    +-----------------------------------+------------------------------------------------------------------------+
    | CAR                               | Bool value enabling common average calculation for g.Nautilus          |
    +-----------------------------------+------------------------------------------------------------------------+
    | AccelerationData                  | Bool value enabling acquisition of acceleration data from g.Nautilus   |
    |                                   | head stage, adds 3 additional channels to the data acquisition for x,  |
    |                                   | y, and z direction                                                     |
    +-----------------------------------+------------------------------------------------------------------------+
    | Counter                           | show a counter as an additional channel                                |
    +-----------------------------------+------------------------------------------------------------------------+
    | LinkQualityInformation            | Bool value enabling additional channel informing about link quality    |
    |                                   | between head stage and base station                                    |
    +-----------------------------------+------------------------------------------------------------------------+
    | BatteryLevel                      | Bool to enable acquisition of additional channel holding information   |
    |                                   | about remaining battery capacity                                       |
    +-----------------------------------+------------------------------------------------------------------------+
    | DigitalIOs                        | Scan the digital channels with the analog inputs and add them as       |
    |                                   | additional channel acquired                                            |
    +-----------------------------------+------------------------------------------------------------------------+
    | ValidationIndicator               | Enables the additional channel validation indicator, informing about   |
    |                                   | the liability of the data recorded                                     |
    +-----------------------------------+------------------------------------------------------------------------+
    | NetworkChannel                    | Unsigned integer value representing the network channel used between   |
    |                                   | head stage and base station, or zero to not change the network channel |
    |                                   |                                                                        |
    |                                   | ATTENTION: ``GetConfiguration()`` will overwrite zeros with the actual |
    |                                   | network channel that the device currently uses                         |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels                          | Array of g.Nautilus channel configurations holding properties for each |
    |                                   | analog channel                                                         |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].ChannelNumber         | Unsigned integer holding the channel number of the analog channel      |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].Enabled               | Bool value selecting the channel for data acquisition                  |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].Sensitivity           | Double value representing the sensitivity of the specified channel     |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].UsedForNoiseReduction | Bool value indicating if channel should be used for noise reduction    |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].UsedForCAR            | Bool value indicating if channel should be used for common average     |
    |                                   | calculation                                                            |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].BandpassFilterIndex   | Perform a digital bandpass filtering of the input channels. Use        |
    |                                   | ``GetBandpassFilters()`` to get filter indices.                        |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].NotchFilterIndex      | Perform a bandstop filtering to suppress the power line frequency of   |
    |                                   | 50 Hz or 60 Hz. Use ``GetNotchFilters()`` to get filter indices.       |
    +-----------------------------------+------------------------------------------------------------------------+
    | Channels[i].BipolarChannel        | Select a zero based channel index as reference channel for an analog   |
    |                                   | channel                                                                |
    +-----------------------------------+------------------------------------------------------------------------+


    Note that some names are unified to work for all devices. See ``name_maps``.
    '''

    def __init__(self, gds_device=None  # (comma-separated list of) device serial/s
                 , exclude_serials=None  # list of serials to exclude
                 , open_exclusively=True # True to prevent a newly created data acquisition session from being opened or accessed by another ``pygds.GDS`` instance with the same list of device serials; otherwise, False
                 , server_ip=SERVER_IP
                 ):
        self.ConfigCount = 0
        self._connectionHandle = NULL
        self._configuration = NULL
        self.Name = b''
        self.Serials = []

        wanted_type = DEVICE_TYPE_NOT_SUPPORTED
        if isinstance(gds_device, int):
            wanted_type = gds_device
        else:
            nlen = len(gds_device) if gds_device else 0
        cd = None
        if nlen <= 1:
            if nlen == 1:
                if gds_device[0] == 'N':
                    wanted_type = DEVICE_TYPE_GNAUTILUS
                elif gds_device[0] == 'H':
                    wanted_type = DEVICE_TYPE_GHIAMP
                elif gds_device[0] == 'U':
                    wanted_type = DEVICE_TYPE_GUSBAMP
            cd = ConnectedDevices(server_ip)
            gds_device = cd.find(wanted_type, exclude_serials)
            assert gds_device != None, "no device found"

        try:
            self.Name = gds_device.encode()
        except:
            self.Name = gds_device

        self.Serials = self.Name.split(b',')
        if self.Serials[0] == b'':
            self.Serials = []

        self.Name = self.Name.decode()
        if self.Name in OpenDevices:
            del OpenDevices[self.Name]

        self._connectionHandle = _ffi.new("GDS_HANDLE *")
        is_creator = _ffi.new("BOOL *")
        if cd is not None:
            serverip, clientip = cd.serverip, cd.clientip
        else:
            serverip, clientip = _server_client(server_ip)
        _chk(_ffi_dll.GDS_Connect((serverip, SERVER_PORT), (clientip, CLIENT_PORT),
                                  self.Serials, len(self.Serials), open_exclusively, self._connectionHandle, is_creator))
        self.IsCreator = is_creator[0]
        assert self._connectionHandle[0] != 0, "device could not be connected"
        OpenDevices[self.Name] = self

        self._serials = []
        self._p_serials = []
        for idev in range(len(self.Serials)):
            sr = self.Serials[idev]
            _serial = _ffi.new("char["+str(DEVICE_NAME_LENGTH_MAX)+"]")
            _ffi.memmove(_serial, sr, len(sr))
            self._serials.append(_serial)
            self._p_serials.append(_ffi.addressof(_serial))

        self._configuration = _ffi.new("GDS_CONFIGURATION_BASE **")
        self._configurationcount = _ffi.new("size_t *")

        self.GetConfiguration()

        self.N_electrodes = sum(sum(x)
                                for x in self.GetAvailableChannels(combine=False))

    def SetConfiguration(self):
        '''GDS.

        ``SetConfiguration()`` needs to be called to send the configuration to the device.

        ``SetConfiguration()`` must not be called if the client is not the session creator and ``self.IsCreator`` equals False
        (i.e. if the constructor of the current ``pygds.GDS`` instance was called with the parameter ``open_exclusively=False``).

        Before calling the underlying ``GDS_SetConfiguration()``, the channels that are not available on the device are removed.
        So one can do ``for ch in d.Channels: ch.Acquire=True`` without the need to consult ``GetAvailableChannels()``.

        '''

        avail = self.GetAvailableChannels()
        for i, c in enumerate(self.Configs):
            for j, ch in enumerate(c.Channels):
                ch.Acquire = ch.Acquire and avail[i][j]
        _chk(_ffi_dll.GDS_SetConfiguration(
            self._connectionHandle[0], self._configuration[0], self.ConfigCount))

    def GetConfiguration(self):
        '''GDS.

        ``GetConfiguration()`` fetches the configuration from the device.
        This is done automatically when instantiating a GDS object.

        '''

        _chk(_ffi_dll.GDS_GetConfiguration(
            self._connectionHandle[0], self._configuration, self._configurationcount))
        assert self._configurationcount[0] != 0, "configuration count is 0"
        self.Configs = []
        for i in range(self._configurationcount[0]):
            fficfg = self._configuration[0][i]
            w = _config_wrap(fficfg)
            self.Configs.append(w)

        self.ConfigCount = self._configurationcount[0]
        if self.ConfigCount >= 1:
            _config_wrap.__init__(self, self._configuration[0][0])

    def GetDataInfo(self,
                    scanCount  # number of scans
                    ):
        '''GDS.

        ``GetDataInfo()`` returns (channelsPerDevice, bufferSizeInSamples).

        *channelsPerDevice* is a list of channels for each device.

        *bufferSizeInSamples* is the total number of samples.

        .. code:: py

            >>> import pygds
            >>> d = pygds.GDS()
            >>> scanCount = 500
            >>> channelsPerDevice, bufferSizeInSamples = d.GetDataInfo(scanCount)
            >>> sum(channelsPerDevice)*scanCount == bufferSizeInSamples
            True
            >>> d.Close(); del d

        '''

        sc = _ffi.new("size_t *")
        sc[0] = scanCount
        channelsPerDeviceCount = _ffi.new("size_t *")
        bufferSizeInSamples = _ffi.new("size_t *")
        _chk(_ffi_dll.GDS_GetDataInfo(
            self._connectionHandle[0], sc, NULL, channelsPerDeviceCount, bufferSizeInSamples))
        channelsPerDevice = _ffi.new("size_t[%s]" % channelsPerDeviceCount[0])
        _chk(_ffi_dll.GDS_GetDataInfo(
            self._connectionHandle[0], sc, channelsPerDevice, channelsPerDeviceCount, bufferSizeInSamples))
        return ([int(x) for x in channelsPerDevice], bufferSizeInSamples[0])

    def N_ch_calc(self):
        '''GDS.

        ``N_ch_calc()`` returns the number of configured channels.
        After the first call, you can use ``d.N_ch`` to get the number of configured channels.

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> n = d.N_ch_calc()
            >>> d.N_ch == n
            True
            >>> d.Close(); del d

        ``d.N_electrodes`` is the number of electrodes in the GDS connection for all connected devices.
        ``d.N_ch`` can be equal, smaller or larger than ``d.N_electrodes``, depending on the configuration.

        '''

        self.N_ch = self.IndexAfter()
        return self.N_ch if self.N_ch else 0

    def NumberOfScans_calc(self):
        '''GDS.

        Sets ``d.NumberOfScans`` by mapping ``d.SamplingRate`` via ``GetSupportedSamplingRates()``.

        '''
        
        sr = self.GetSupportedSamplingRates()
        for i, c in enumerate(self.Configs):
            c.NumberOfScans = sr[i].get(c.SamplingRate, 8)

    def IndexAfter(self,
                   chname=''  # '1',..,'Counter',... => index after; '' => channel count
                   ):
        '''GDS.

        Get the zero-based channel index one position after the one-based ``chname``.
        ``chname`` can also be one of::

                Counter
                Trigger

        and for g.Nautilus also::

                AccelerationData
                LinkQualityInformation
                BatteryLevel
                DigitalIOs
                ValidationIndicator

        Without ``chname`` it gives the count of configured channels.

        For more devices per GDS object one can use::

            name+serial, e.g. 1UB-2008.07.01

        to get the index of a channel of a specific device.

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> d.IndexAfter('4'+d.Name)
            4
            >>> d.IndexAfter('4')
            4
            >>> d.IndexAfter('AccelerationData')>=0
            True
            >>> d.IndexAfter('Counter')>=0
            True
            >>> d.IndexAfter('LinkQualityInformation')>=0
            True
            >>> d.IndexAfter('BatteryLevel')>=0
            True
            >>> d.IndexAfter('DigitalIOs')>=0
            True
            >>> d.IndexAfter('Trigger')>=0
            True
            >>> d.IndexAfter('ValidationIndicator')>=0
            True
            >>> d.IndexAfter('')==d.N_ch_calc()
            True
            >>> d.Close(); del d

        '''

        try:
            chn = str(chname)
        except:
            chn = chname
        avail = self.GetAvailableChannels()
        idx = 0
        for idev in range(self.ConfigCount):
            serial = self.Serials[idev].decode()
            lenavail = len(avail[idev])

            def isok(s):
                return chn and (chn in s or (s+serial).startswith(chn))
            dt = self.Configs[idev].DeviceType
            if dt == DEVICE_TYPE_GNAUTILUS:
                for i in range(lenavail):
                    if avail[idev][i]:
                        idx = idx+1
                    if isok(str(i+1)):
                        return idx
                if self.Configs[idev].AccelerationData:
                    idx = idx+3
                if isok('AccelerationData'):
                    return idx
                if self.Configs[idev].Counter:
                    idx = idx+1
                if isok('Counter'):
                    return idx
                if self.Configs[idev].LinkQualityInformation:
                    idx = idx+1
                if isok('LinkQualityInformation'):
                    return idx
                if self.Configs[idev].BatteryLevel:
                    idx = idx+1
                if isok('BatteryLevel'):
                    return idx
                if self.Configs[idev].DigitalIOs:
                    idx = idx+1
                if isok('DigitalIOs'):
                    return idx
                if isok('Trigger'):
                    return idx
                if isok('DI'):
                    return idx
                if self.Configs[idev].ValidationIndicator:
                    idx = idx+1
                if isok('ValidationIndicator'):
                    return idx
            elif dt == DEVICE_TYPE_GHIAMP:
                if isok('Counter') and avail[idev][0]:
                    return idx+1  # first channel becomes counter
                for i in range(lenavail):
                    if avail[idev][i]:
                        idx = idx+1
                    if isok(str(i+1)):
                        return idx
                if self.Configs[idev].Trigger:
                    idx = idx+1
                if isok('Trigger'):
                    return idx
                if isok('DI'):
                    return idx
            elif dt == DEVICE_TYPE_GUSBAMP:
                for i in range(lenavail):
                    if avail[idev][i]:
                        idx = idx+1
                    if isok(str(i+1)):
                        return idx
                if self.Configs[idev].Counter and isok('Counter'):
                    return idx  # last channel becomes counter
                if self.Configs[idev].Trigger:
                    idx = idx+1
                if isok('Trigger'):
                    return idx
                if isok('DI'):
                    return idx
        return idx

    def GetData(self,
                # number of scans. A scan is a sample for each channel.
                scanCount,
                more=None # a function that takes the samples and must return True if more samples are wanted
                ):
        '''GDS.

        ``GetData()`` gets the data from the device.

        It allocates ``scanCount*N_ch*4`` memory two times and fills one copy 
        in a separate thread with sample data from the device, while the other 
        copy is processed by the ``more`` function in the current thread.
        Then it swaps the two buffers.

        ``more(samples)`` is an optional callback handler function that is called on data retrieval with the current samples as parameter.
        The user can decide by its return value whether to continue acquisition by returning True.

        ``more`` must copy the samples to reuse them later.
        
        It is recommended to call ``GetData()`` only once per acquisition session, especially for data acquisition of longer or yet unknown duration.
        ``GetData()`` internally acquires data periodically in small blocks from the device, depending on the configured ``d.NumberOfScans`` parameter.
        To process data periodically in blocks, provide a ``more`` callback handler function that returns True until data acquisition should stop.
        Set the ``scanCount`` parameter to the block size to process at once, typically a smaller value that results in about 30 milliseconds of data
        at the configured sampling rate. This way, the callback handler function gets called periodically and data can be processed within its body.
        
        If regular timing is crucial, ensure that the ``scanCount`` parameter is an integral multiple of the 
        device's acquisition block size configured by the device configuration's ``NumberOfScans`` parameter. 
        Otherwise, the ``more`` callback will deliver data at irregular intervals.

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> samples = []
            >>> more = lambda s: samples.append(s.copy()) or len(samples)<2
            >>> data=d.GetData(d.SamplingRate, more)
            >>> len(samples)
            2
            >>> d.Close(); del d

        '''

        assert scanCount > 0, "No scans were requested"
        self.N_ch_calc()
        assert self.N_ch > 0, "No channels were configured"
        size = scanCount * self.N_ch

        self.do = Queue()
        self.done = Queue()
        self.thread = Thread(target=self._getdata)
        self.thread.start()
        try:
            samples = np.zeros((scanCount, self.N_ch), dtype=np.float32)
            if more:
                samples1 = np.zeros((scanCount, self.N_ch), dtype=np.float32)
                smpls = [samples, samples1]
            else:
                smpls = [samples]
            s = None
            i = -1
            while True:
                i = (i+1) % 2
                self.do.put((scanCount, size, smpls[i]))
                if not isinstance(s, type(None)) and more != None:
                    if not more(s):
                        break
                s,loops,timeout = self.done.get()
                if timeout:
                    raise RuntimeError('Timeout occurred. No data after {0} s with {1} retries'.format(AcquisitionTimeout, loops))
                if more == None:
                    break
            self.do.put((0, 0, None))
            self.do = None
            self.done = None
        finally:
            self.thread.join()
            self.thread = None
        return s

    def _getdata(self):
        if self.IsCreator:
            _chk(_ffi_dll.GDS_StartAcquisition(self._connectionHandle[0]))
        _chk(_ffi_dll.GDS_StartStreaming(self._connectionHandle[0]))
        timeoutOccurred = False
        try:
            while not timeoutOccurred:
                scanCount, size, samples = self.do.get()
                if scanCount == 0:
                    break
                sc = _ffi.new("size_t *")
                scantotal = int(0)
                loops = 0
                start = timer()
                resetTimer = True
                while self.do and scantotal < scanCount and not timeoutOccurred:
                    sc[0] = 0
                    offset = scantotal*self.N_ch
                    left = size - offset
                    fa = _ffi.cast("float *", samples.ctypes.data+offset*4)
                    if resetTimer:
                        start = timer()
                    _chk(_ffi_dll.GDS_GetData(
                        self._connectionHandle[0], sc, fa, left))
                    if sc[0] > 0:
                        scantotal += sc[0]
                        resetTimer = True
                        loops = 0
                    else:
                        loops = loops + 1
                        resetTimer = False
                        now = timer()
                        if now - start > AcquisitionTimeout:
                            timeoutOccurred = True
                            break
                self.done.put((samples,loops,timeoutOccurred))
        except:
            pass
        finally:
            _chk(_ffi_dll.GDS_StopStreaming(self._connectionHandle[0]))
            if self.IsCreator:
                _chk(_ffi_dll.GDS_StopAcquisition(self._connectionHandle[0]))

    def GetAvailableChannels(self,
                             combine=True  # and-combine the C API's GDS_XXX_GetAvailableChannels with the channel's Acquire flag
                             ):
        '''GDS.

        ``GetAvailableChannels()`` wraps C API's ``GDS_XXX_GetAvailableChannels()``.
        The return value of each device is an entry in the returned list.
        ``d.GetAvailableChannels()[0]`` is a list of 0 or 1.

        This is called when instantiating a GDS object to initialize the ``N_electrodes`` member.
        It is also called in ``SetConfiguration()`` to ignore the channels that are not available.
        And it is called in ``IndexAfter()`` and thus also in
        ``N_ch_calc()`` to get the channel index or the configured channel count.
        There should be no reason to call this directly.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("BOOL a["+str(GNAUTILUS_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GNAUTILUS_GetAvailableChannels(
                    self._connectionHandle[0], pserial, pa))
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("BOOL a["+str(GHIAMP_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GHIAMP_GetAvailableChannels(
                    self._connectionHandle[0], pserial, pa))
            elif dt == DEVICE_TYPE_GUSBAMP:
                a = [1]*GUSBAMP_CHANNELS_MAX
            if combine:
                res.append([x and c.Acquire for x, c in zip(
                    a, self.Configs[idev].Channels)])
            else:
                res.append(a)
        return res

    def GetAvailableDigitalIOs(self):
        '''GDS.

        ``GetAvailableDigitalIOs()`` wraps the g.Nautilus' ``GDS_GNAUTILUS_GetAvailableDigitalIOs()``.
        g.Nautilus only.

        The return value of each device is an entry in the returned list.
        ``d.GetAvailableDigitalIOs()[0]`` is a list of dicts, each with these keys:

        +---------------+--------------------------------------------------------------------------+
        | ChannelNumber | Unsigned integer representing the digital IO number                      |
        +---------------+--------------------------------------------------------------------------+
        | Direction     | String representing the direction of the digital channel (In=0 or Out=1) |
        +---------------+--------------------------------------------------------------------------+

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("GDS_GNAUTILUS_DIGITAL_IO_CHANNEL ["+str(8)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GNAUTILUS_GetAvailableDigitalIOs(
                    self._connectionHandle[0], pserial, pa))
                res.append([_ffi_struct_wrap(aa)._to_python() for aa in a])
        return res

    def GetAsyncDigitalIOs(self):
        '''GDS.

        ``GetAsyncDigitalIOs()`` wraps the g.USBamp's ``GDS_GUSBAMP_GetAsyncDigitalIOs()``.
        g.USBamp only.

        The return value of each device is an entry in the returned list.
        ``d.GetAsyncDigitalIOs()[0]`` is a list of dicts, each with these keys:

        +---------------+--------------------------------------------------------------+
        | ChannelNumber | Integer value representing the digital channel number        |
        +---------------+--------------------------------------------------------------+
        | Direction     | String holding the digital channel direction (In=0 or Out=1) |
        +---------------+--------------------------------------------------------------+
        | Value         | Current value of the digital channel (true or false)         |
        +---------------+--------------------------------------------------------------+

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                a = _ffi.new(
                    "GDS_GUSBAMP_ASYNC_DIGITAL_IO_CHANNEL ["+str(4)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GUSBAMP_GetAsyncDigitalIOs(
                    self._connectionHandle[0], pserial, pa))
                res.append([_ffi_struct_wrap(aa)._to_python() for aa in a])
        return res

    def SetAsyncDigitalOutputs(self, outputs):
        '''GDS.

        ``SetAsyncDigitalOutputs()`` wraps the g.USBamp ``GDS_GUSBAMP_SetAsyncDigitalOutputs()``.
        g.USBamp only.

        '''

        for idev in range(self.ConfigCount):
            out = outputs[idev]
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                _chk(_ffi_dll.GDS_GUSBAMP_SetAsyncDigitalOutputs(
                    self._connectionHandle[0], pserial, out, len(out)))

    def GetDeviceInformation(self):
        '''GDS.

        ``GetDeviceInformation()`` wraps the C API's ``GDS_XXX_GetDeviceInformation()`` functions.

        The device information for each device is a string entry in the returned list.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                a = _ffi.new(
                    "char ["+str(GUSBAMP_DEVICE_INFORMATION_LENGTH_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GUSBAMP_GetDeviceInformation(
                    self._connectionHandle[0], pserial, pa))
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new(
                    "char ["+str(GHIAMP_DEVICE_INFORMATION_LENGTH_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GHIAMP_GetDeviceInformation(
                    self._connectionHandle[0], pserial, pa))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new(
                    "char ["+str(GNAUTILUS_DEVICE_INFORMATION_LENGTH_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GNAUTILUS_GetDeviceInformation(
                    self._connectionHandle[0], pserial, pa))
            res.append(_ffi.string(a).decode())
        return res

    def GetImpedance(self,
                     # list of bool or int telling which electrode is active (g.HIamp only)
                     active=None
                     ):
        '''GDS.

        **This method is deprecated** (use ``GetImpedanceEx()`` instead).

        ``GetImpedance()`` wraps the C API's ``GDS_XXX_GetImpedance()`` functions.

        Gets the impedances for all channels of all devices.
        The impedances of each device are a list entry in the returned list.

        Note, that for g.Nautilus electrode Cz must be connected to GND,
        else an exception occurs.

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> imps = d.GetImpedance([1]*len(d.Channels))
            >>> len(imps[0])==len(d.Channels)
            True
            >>> d.Close(); del d

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                N = GUSBAMP_CHANNELS_MAX
                a = [0]*N
                pa = _ffi.new("double *")
                for i in range(N):
                    _chk(_ffi_dll.GDS_GUSBAMP_GetImpedance(
                        self._connectionHandle[0], pserial, i+1, pa))
                    a[i] = pa[0]
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("double ["+str(GHIAMP_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                b = _ffi.new("BOOL ["+str(GHIAMP_CHANNELS_MAX)+"]")
                if active and isinstance(active, list):
                    for i in range(GHIAMP_CHANNELS_MAX):
                        b[i] = active[i]
                pb = _ffi.addressof(b)
                _chk(_ffi_dll.GDS_GHIAMP_GetImpedance(
                    self._connectionHandle[0], pserial, pb, pa))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("double ["+str(GNAUTILUS_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GNAUTILUS_GetImpedance(
                    self._connectionHandle[0], pserial, pa))
            res.append([x for x in a])
        return res

    def GetImpedanceEx(self,
                       # the type of connected electrodes, can be 0=Passive, 1=Active. See ``pygds.ELECTRODE_TYPE_XXX`` constants. (g.HIamp only)
                       electrodeType=0,
                       # True if a TMS electrode connector box is connected and impedance values of all channels should be corrected by 10 kilo-ohm; otherwise, False. (g.HIamp only)
                       tmsBoxCorrection=False,
                       # If this is the start of a new measurement session (i.e. the first in a loop of consecutive calls to ``GetImpedanceEx()``), set this flag to True to clear internal interim values of previous measurements; for the following repeated calls, set this flag to False until the end of the measurement session. (g.HIamp only)
                       firstMeasurementOfSession=True
                       ):
        '''GDS.

        ``GetImpedanceEx()`` wraps the C API's ``GDS_XXX_GetImpedanceEx()`` functions.

        Gets the impedances for all channels of all devices.
        The impedances of each device are a list entry in the returned list.
        Negative values represent different error codes.
        See the C or .NET Client API manuals for details.

        Note, that for g.Nautilus electrode Cz must be connected to GND,
        else an exception occurs.

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> imps = d.GetImpedanceEx(pygds.ELECTRODE_TYPE_PASSIVE, False, True)
            >>> len(imps[0])==len(d.Channels)
            True
            >>> d.Close(); del d

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                N = GUSBAMP_CHANNELS_MAX + GUSBAMP_GROUPS_MAX
                a = [0]*N
                pa = _ffi.new("double *")
                for i in range(N):
                    _chk(_ffi_dll.GDS_GUSBAMP_GetImpedanceEx(
                        self._connectionHandle[0], pserial, i+1, pa))
                    a[i] = pa[0]
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("double ["+str(GHIAMP_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GHIAMP_GetImpedanceEx(
                    self._connectionHandle[0], pserial, electrodeType, tmsBoxCorrection, firstMeasurementOfSession, pa))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("double ["+str(GNAUTILUS_CHANNELS_MAX)+"]")
                pa = _ffi.addressof(a)
                _chk(_ffi_dll.GDS_GNAUTILUS_GetImpedanceEx(
                    self._connectionHandle[0], pserial, pa))
            res.append([x for x in a])
        return res

    def GetScaling(self):
        '''GDS.

        ``GetScaling()`` wraps the C API's ``GDS_XXX_GetScaling()`` functions.

        The return value of each device is a dict entry in the returned list.
        Each dict has the fields:

        +--------+-------------------------------------------------------------------------------+
        | Factor | Array holding single type values with scaling factor for each analog channel. |
        +--------+-------------------------------------------------------------------------------+
        | Offset | Array holding single type values with offset for each analog channel.         |
        +--------+-------------------------------------------------------------------------------+

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                a = _ffi.new("GDS_GUSBAMP_SCALING *")
                _chk(_ffi_dll.GDS_GUSBAMP_GetScaling(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("GDS_GHIAMP_SCALING *")
                _chk(_ffi_dll.GDS_GHIAMP_GetScaling(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("GDS_GNAUTILUS_SCALING *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetScaling(
                    self._connectionHandle[0], pserial, a))
            if dt == DEVICE_TYPE_GUSBAMP:
                name_map = name_maps.get('GDS_GUSBAMP_SCALING', {})
            elif dt == DEVICE_TYPE_GHIAMP:
                name_map = name_maps.get('GDS_GHIAMP_SCALING', {})
            elif dt == DEVICE_TYPE_GNAUTILUS:
                name_map = name_maps.get('GDS_GNAUTILUS_SCALING', {})
            res.append(_ffi_struct_wrap(a, name_map=name_map))
        return res

    def Calibrate(self):
        '''GDS.

        ``Calibrate()`` wraps the C API's ``GDS_XXX_Calibrate()`` functions,
        which calibrates the device.

        The return value of each device is a dict entry in the returned list.
        ``d.Calibrate()[0]`` is a dict with these keys:

        +---------------+-------------------------------------------------------------------------------+
        | ScalingFactor | Array holding single type values with scaling factor for each analog channel. |
        +---------------+-------------------------------------------------------------------------------+
        | Offset        | Array holding single type values with offset for each analog channel.         |
        +---------------+-------------------------------------------------------------------------------+

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                a = _ffi.new("GDS_GUSBAMP_SCALING *")
                _chk(_ffi_dll.GDS_GUSBAMP_Calibrate(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("GDS_GHIAMP_SCALING *")
                _chk(_ffi_dll.GDS_GHIAMP_Calibrate(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("GDS_GNAUTILUS_SCALING *")
                _chk(_ffi_dll.GDS_GNAUTILUS_Calibrate(
                    self._connectionHandle[0], pserial, a))
            if dt == DEVICE_TYPE_GUSBAMP:
                name_map = name_maps.get('GDS_GUSBAMP_SCALING', {})
            elif dt == DEVICE_TYPE_GHIAMP:
                name_map = name_maps.get('GDS_GHIAMP_SCALING', {})
            elif dt == DEVICE_TYPE_GNAUTILUS:
                name_map = name_maps.get('GDS_GNAUTILUS_SCALING', {})
            res.append(_ffi_struct_wrap(a, name_map=name_map))
        return res

    def SetScaling(self,
                   scaling  # an array of GDS_XXX_SCALING structs or equivalent dicts
                   ):
        '''GDS.

        ``SetScaling()`` wraps the C API's ``GDS_XXX_SetScaling()`` functions.

        ``SetScaling()`` sets the scaling on the device.

        '''

        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            try:
                a = scaling[idev]._ffi_struct
            except:
                a = scaling[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                _chk(_ffi_dll.GDS_GUSBAMP_SetScaling(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GHIAMP:
                _chk(_ffi_dll.GDS_GHIAMP_SetScaling(
                    self._connectionHandle[0], pserial, a))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                _chk(_ffi_dll.GDS_GNAUTILUS_SetScaling(
                    self._connectionHandle[0], pserial, a))

    def ResetScaling(self):
        '''GDS.

        ``ResetScaling()`` wraps the g.Nautilus ``GDS_GNAUTILUS_ResetScaling()`` function.

        The scaling is reset to Offset=0.0 and Factor=1.0. g.Nautilus only.

        '''

        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                _chk(_ffi_dll.GDS_GNAUTILUS_ResetScaling(
                    self._connectionHandle[0], pserial))

    def GetNetworkChannel(self):
        '''GDS.

        ``GetNetworkChannel()`` wraps the C API's ``GDS_GNAUTILUS_GetNetworkChannel()``.

        The currently used g.Nautilus network channel is an entry in the returned list.
        The device configuration stored behind in ``self.Configs`` and ``self.NetworkChannel``,
        respectively, is also updated with the retrieved network channel.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                a = _ffi.new("uint32_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetNetworkChannel(
                    self._connectionHandle[0], pserial, a))
                networkchannel = a[0]
                res.append(networkchannel)
                self.Configs[idev].NetworkChannel = networkchannel
        return res

    def GetFactoryScaling(self):
        '''GDS.

        ``GetFactoryScaling()`` wraps C API's ``GDS_GHIAMP_GetFactoryScaling()``.

        The factory scaling is an entry for each g.HIamp in the returned list.
        Only g.HIamp.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GHIAMP:
                a = _ffi.new("GDS_GHIAMP_SCALING *")
                _chk(_ffi_dll.GDS_GHIAMP_GetFactoryScaling(
                    self._connectionHandle[0], pserial, a))
                res.append(_ffi_struct_wrap(
                    a, name_map=name_maps.get('GDS_GHIAMP_SCALING', {})))
        return res

    #def GetTemperature(self):
    #    '''GDS.
    #    The temperature for each device is an entry in the returned list.
    #    '''
    #    res = []
    #    for idev in range(self.ConfigCount):
    #        dt = self.Configs[idev].DeviceType
    #        pserial = self._p_serials[idev]
    #        a = _ffi.new("float *")
    #        if dt == DEVICE_TYPE_GUSBAMP:
    #            _chk(_ffi_dll.GDS_GUSBAMP_GetTemperature(
    #                self._connectionHandle[0], pserial, a))
    #        elif dt == DEVICE_TYPE_GHIAMP:
    #            _chk(_ffi_dll.GDS_GHIAMP_GetTemperature(
    #                self._connectionHandle[0], pserial, a))
    #        elif dt == DEVICE_TYPE_GNAUTILUS:
    #            InputSignal = self.Configs[idev].InputSignal
    #            self.Configs[idev].InputSignal = GNAUTILUS_INPUT_TEMPERATURE
    #            Sensitivity = [0]*GNAUTILUS_CHANNELS_MAX
    #            for j in range(GNAUTILUS_CHANNELS_MAX):
    #                Sensitivity[j] = self.Configs[idev].Channels[j].Sensitivity
    #                self.Configs[idev].Channels[j].Sensitivity = 375000
    #            try:
    #                self.SetConfiguration()
    #                scanCount = self.Configs[idev].SamplingRate
    #                tmps = self.GetData(scanCount)
    #                tmps = (tmps-145300) / 490 + 25
    #                tmps = tmps[tmps.shape[0]//2:, :]
    #                tmp = tmps[tmps > 0].mean()
    #                a = [tmp]
    #            finally:
    #                self.Configs[idev].InputSignal = InputSignal
    #                for j in range(GNAUTILUS_CHANNELS_MAX):
    #                    self.Configs[idev].Channels[j].Sensitivity = Sensitivity[j]
    #                self.SetConfiguration()
    #        res.append(a[0])
    #    return res

    def GetSupportedSamplingRates(self):
        '''GDS.

        ``GetSupportedSamplingRates()`` wraps the C API's ``GDS_XXX_GetSupportedSamplingRates()`` functions.

        For each device a dict ``{SamplingRate:NumberOfScans}`` is an entry in the returned list.

        You can do ``d.NumberOfScans=d.GetSupportedSamplingRates()[0][d.SamplingRate]``
        to set the recommended NumberOfScans.
        This is done when using ``d.NumberOfScans_calc()``, and if there are more devices per GDS object.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GUSBAMP_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new(
                    "GDS_GUSBAMP_SAMPLING_RATE_FEATURES["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GUSBAMP_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, a, sz))
                res.append(dict(sorted(_ffi_struct_wrap(
                    x)._to_python().values(), reverse=True) for x in a))
            elif dt == DEVICE_TYPE_GHIAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GHIAMP_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new(
                    "GDS_GHIAMP_SAMPLING_RATE_FEATURES["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GHIAMP_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, a, sz))
                res.append(dict(sorted(_ffi_struct_wrap(
                    x)._to_python().values(), reverse=True) for x in a))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("uint32_t["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedSamplingRates(
                    self._connectionHandle[0], pserial, a, sz))
                numberofscans = {250: 8, 500: 15}
                # 4000 is only internal for impedance measurement
                res.append({int(x): numberofscans.get(x, 8)
                            for x in a if x != 4000})
        return res

    def GetBandpassFilters(self):
        '''GDS.

        ``GetBandpassFilters()`` wraps the C API's ``GDS_XXX_GetBandpassFilters()`` functions.

        In the returned list an entry per device is a list of dicts, with one dict for each filter.
        The dicts also contain the key ``BandpassFilterIndex`` to be used to set the filter.

        The fields per filter are:

        +----------------------+----------------------------------------------------------------------+
        | BandpassFilterIndex  | Use this for the according channel field                             |
        +----------------------+----------------------------------------------------------------------+
        | SamplingRate         | Double value holding the sampling rate for which the filter is valid |
        +----------------------+----------------------------------------------------------------------+
        | Order                | Unsigned integer holding filter order                                |
        +----------------------+----------------------------------------------------------------------+
        | LowerCutoffFrequency | Double representing lower cutoff frequency of the filter             |
        +----------------------+----------------------------------------------------------------------+
        | UpperCutoffFrequency | Double representing upper cutoff frequency of the filter             |
        +----------------------+----------------------------------------------------------------------+
        | TypeId               | Representing type of filter                                          |
        +----------------------+----------------------------------------------------------------------+

        To choose a filter for the desired sampling rate, you can do this:

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> f_s_2 = sorted(d.GetSupportedSamplingRates()[0].items())[1] #512 or 500
            >>> d.SamplingRate, d.NumberOfScans = f_s_2
            >>> BP = [x for x in d.GetBandpassFilters()[0] if x['SamplingRate'] == d.SamplingRate]
            >>> for ch in d.Channels:
            ...     ch.Acquire = True
            ...     if BP:
            ...         ch.BandpassFilterIndex = BP[0]['BandpassFilterIndex']
            >>> d.SetConfiguration()
            >>> d.GetData(d.SamplingRate).shape[0] == d.SamplingRate
            True
            >>> d.Close(); del d

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GUSBAMP_GetBandpassFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GUSBAMP_GetBandpassFilters(
                    self._connectionHandle[0], pserial, a, sz))
            elif dt == DEVICE_TYPE_GHIAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GHIAMP_GetBandpassFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GHIAMP_GetBandpassFilters(
                    self._connectionHandle[0], pserial, a, sz))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetBandpassFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetBandpassFilters(
                    self._connectionHandle[0], pserial, a, sz))

            def update(x, d): return x.update(d) or x
            res.append([update(_ffi_struct_wrap(x)._to_python(), {
                       'BandpassFilterIndex': i}) for i, x in enumerate(a)])
        return res

    def GetNotchFilters(self):
        '''GDS.

        ``GetNotchFilters()`` wraps the C API's ``GDS_XXX_GetNotchFilters()`` functions.

        In the returned list an entry per device is a list of dicts, with one dict for each filter.
        The dicts also contain the key ``NotchFilterIndex`` to be used to set the filter.

        The fields per filter are:

        +----------------------+----------------------------------------------------------------------+
        | NotchFilterIndex     | Use this for the according channel field                             |
        +----------------------+----------------------------------------------------------------------+
        | SamplingRate         | Double value holding the sampling rate for which the filter is valid |
        +----------------------+----------------------------------------------------------------------+
        | Order                | Unsigned integer holding filter order                                |
        +----------------------+----------------------------------------------------------------------+
        | LowerCutoffFrequency | Double representing lower cutoff frequency of the filter             |
        +----------------------+----------------------------------------------------------------------+
        | UpperCutoffFrequency | Double representing upper cutoff frequency of the filter             |
        +----------------------+----------------------------------------------------------------------+
        | TypeId               | Representing type of filter                                          |
        +----------------------+----------------------------------------------------------------------+

        To choose a filter for the desired sampling rate you can do this:

        .. code:: py

            >>> import pygds; d = pygds.GDS()
            >>> f_s_2 = sorted(d.GetSupportedSamplingRates()[0].items())[1] #512 or 500
            >>> d.SamplingRate, d.NumberOfScans = f_s_2
            >>> N = [x for x in d.GetNotchFilters()[0] if x['SamplingRate'] == d.SamplingRate]
            >>> for ch in d.Channels:
            ...     ch.Acquire = True
            ...     if N:
            ...         ch.NotchFilterIndex = N[0]['NotchFilterIndex']
            >>> d.SetConfiguration()
            >>> d.GetData(d.SamplingRate).shape[0] == d.SamplingRate
            True
            >>> d.Close(); del d

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GUSBAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GUSBAMP_GetNotchFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GUSBAMP_GetNotchFilters(
                    self._connectionHandle[0], pserial, a, sz))
            elif dt == DEVICE_TYPE_GHIAMP:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GHIAMP_GetNotchFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GHIAMP_GetNotchFilters(
                    self._connectionHandle[0], pserial, a, sz))
            elif dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetNotchFilters(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_FILTER_INFO["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetNotchFilters(
                    self._connectionHandle[0], pserial, a, sz))

            def update(x, d): return x.update(d) or x
            res.append([update(_ffi_struct_wrap(x)._to_python(), {
                       'NotchFilterIndex': i}) for i, x in enumerate(a)])
        return res

    def GetSupportedSensitivities(self):
        '''GDS.

        ``GetSupportedSensitivities()`` wraps the C API's ``GDS_GNAUTILUS_GetSupportedSensitivities()``.

        The supported sensitivities for each g.Nautilus device are an entry in the returned list.
        g.Nautilus only.

        ``d.GetSupportedSensitivities()[0]`` is a list of integers.
        Each integer can be used as the channel's sesnsitivity.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedSensitivities(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("double["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedSensitivities(
                    self._connectionHandle[0], pserial, a, sz))
                res.append([x for x in a])
        return res

    def GetSupportedNetworkChannels(self):
        '''GDS.

        ``GetSupportedNetworkChannels()`` wraps C API's ``GDS_GNAUTILUS_GetSupportedNetworkChannels()``.

        The supported network channels for each g.Nautilus device are an entry in the returned list.
        g.Nautilus only.

        ``GetSupportedNetworkChannels()[0]`` is a list of integers.
        Each integer can be used in ``d.SetNetworkChannel()``.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedNetworkChannels(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("uint32_t["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedNetworkChannels(
                    self._connectionHandle[0], pserial, a, sz))
                res.append([int(x) for x in a])
        return res

    def GetSupportedInputSources(self):
        '''GDS.

        ``GetSupportedInputSources()`` function wraps ``GDS_GNAUTILUS_GetSupportedInputSources()``.

        The supported g.Nautilus input sources for each g.Nautilus device are an entry in the returned list.
        g.Nautilus only.

        ``d.GetSupportedInputSources()[0]`` is a list of
        integers corresponding to the ``pygds.GDS_GNAUTILUS_INPUT_XXX`` constants.
        Each integer can be used for ``d.InputSignal``.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedInputSources(
                    self._connectionHandle[0], pserial, NULL, sz))
                a = _ffi.new("GDS_GNAUTILUS_INPUT_SIGNAL["+str(sz[0])+"]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetSupportedInputSources(
                    self._connectionHandle[0], pserial, a, sz))
                res.append([int(x) for x in a])
        return res

    def GetChannelNames(self):
        '''GDS.

        ``GetChannelNames()`` wraps C API's ``GDS_GNAUTILUS_GetChannelNames()``.

        A list of channel names for each g.Nautilus device is an entry in the returned list.
        g.Nautilus only.

        ``d.GetChannelNames()[0]`` is a list of strings.
        The strings correspond to the labels on the electrodes.

        '''

        res = []
        for idev in range(self.ConfigCount):
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                mountedModulesCount = _ffi.new("uint32_t *")
                sz = _ffi.new("size_t *")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetChannelNames(
                    self._connectionHandle[0], pserial, mountedModulesCount, NULL, sz))
                a = _ffi.new("char["+str(sz[0])+"][4]")
                _chk(_ffi_dll.GDS_GNAUTILUS_GetChannelNames(
                    self._connectionHandle[0], pserial, mountedModulesCount, a, sz))
                res.append([_ffi.string(aa).decode() for aa in a])
        return res

    def SetNetworkChannel(self,
                          networkchannels  # integer, or list of integers in case of more attached devices
                          ):
        '''GDS.

        ``SetNetworkChannel()`` wraps the C API's ``GDS_GNAUTILUS_SetNetworkChannel()``.
        g.Nautilus only.

        ``SetNetworkChannel()`` sets the g.Nautilus network channel.
        The device configuration stored behind in ``self.Configs`` and 
        ``self.NetworkChannel``, respectively, is also updated with the
        new network channel.

        *networkchannels* is one of the integers returned by GetSupportedNetworkChannels().
        '''

        for idev in range(self.ConfigCount):
            try:
                networkchannel = networkchannels[idev]
            except:
                networkchannel = networkchannels
            dt = self.Configs[idev].DeviceType
            pserial = self._p_serials[idev]
            if dt == DEVICE_TYPE_GNAUTILUS:
                _chk(_ffi_dll.GDS_GNAUTILUS_SetNetworkChannel(
                    self._connectionHandle[0], pserial, networkchannel))
                self.Configs[idev].NetworkChannel = networkchannel

    def Close(self):
        '''GDS.

        Closes the device.

        All GDS objects are removed automatically when exiting Python.

        To remove a GDS object manually, use::
        
        .. code:: py
        
            d.Close()
            del d

        '''

        try:
            del OpenDevices[self.Name]
        except:
            pass

        if self._configuration == NULL:
            return

        if self._connectionHandle[0] != -1:
            _chk(_ffi_dll.GDS_Disconnect(self._connectionHandle))

        if self._configuration[0] != -1:
            _chk(_ffi_dll.GDS_FreeConfigurationList(
                self._configuration, self.ConfigCount))

        self._configuration = NULL
        self._connectionHandle = NULL

        self.Name = None
        self.Serials = []
        self._serials = []
        self._p_serials = []

    def __del__(self):
        self.Close()

    def __str__(self):
        return b','.join(self.Serials).decode()+"\n\n"+"\n\n".join([str(c) for c in self.Configs])


'''

Demo Code
---------

'''


def configure_demo(d, testsignal=False, acquire=1):
    '''
    Makes a configuration for the demos.

    The device configuration fields are members of the device object d.
    If d.ConfigCount>1, i.e. more devices are connected, use d.Configs[i] instead.

    Config names are unified: See ``name_maps``.

    This does not configure a filter.
    Note that g.HIamp version < 1.0.9 will have wrong first value without filters.

    '''
    if d.DeviceType == DEVICE_TYPE_GNAUTILUS:
        sensitivities = d.GetSupportedSensitivities()[0]
        d.SamplingRate = 250
        if testsignal:
            d.InputSignal = GNAUTILUS_INPUT_SIGNAL_TEST_SIGNAL
        else:
            d.InputSignal = GNAUTILUS_INPUT_SIGNAL_ELECTRODE
    else:
        d.SamplingRate = 256
        d.InternalSignalGenerator.Enabled = testsignal
        d.InternalSignalGenerator.Frequency = 10
    d.NumberOfScans_calc()
    d.Counter = 0
    d.Trigger = 0
    for i, ch in enumerate(d.Channels):
        ch.Acquire = acquire
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex = -1
        ch.BipolarChannel = 0  # 0 => to GND
        if d.DeviceType == DEVICE_TYPE_GNAUTILUS:
            ch.BipolarChannel = -1  # -1 => to GND
            ch.Sensitivity = sensitivities[5]
            ch.UsedForNoiseReduction = 0
            ch.UsedForCAR = 0
    #not unified
    if d.DeviceType == DEVICE_TYPE_GUSBAMP:
        d.ShortCutEnabled = 0
        d.CommonGround = [1]*4
        d.CommonReference = [1]*4
        d.InternalSignalGenerator.WaveShape = GUSBAMP_WAVESHAPE_SINE
        d.InternalSignalGenerator.Amplitude = 200
        d.InternalSignalGenerator.Offset = 0
    elif d.DeviceType == DEVICE_TYPE_GHIAMP:
        d.HoldEnabled = 0
    elif d.DeviceType == DEVICE_TYPE_GNAUTILUS:
        d.NoiseReduction = 0
        d.CAR = 0
        d.ValidationIndicator = 1
        d.AccelerationData = 1
        d.LinkQualityInformation = 1
        d.BatteryLevel = 1


def demo_counter():
    '''
    This demo:

    - configures to internal test signal
    - records 1 second
    - displays the counter
    - displays channel 2

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # configure
    configure_demo(d, testsignal=d.DeviceType != DEVICE_TYPE_GUSBAMP)
    d.Counter = 1
    # set configuration
    d.SetConfiguration()
    # get data
    data = d.GetData(d.SamplingRate)
    # plot counter
    scope = Scope(1/d.SamplingRate, modal=True, ylabel='n',
                  xlabel='t/s', title='Counter')
    icounter = d.IndexAfter('Counter')-1
    scope(data[:, icounter:icounter+1])
    plt.show()
    # plot second channel
    scope = Scope(1/d.SamplingRate, modal=True, ylabel=u'U/μV',
                  xlabel='t/s', title='Channel 2')
    scope(data[:, 1:2])
    # or
    # plt.plot(data[1:,1])
    #plt.title('Channel 2')
    plt.show()
    # close
    d.Close()
    del d


def demo_save():
    '''
    This demo:

    - records the internal test signal
    - saves the acquired data after recording

    Have a device:

    - connected to the PC and
    - switched on

    '''
    filename = 'demo_save.npy'
    assert not os.path.exists(
        filename), "the file %s must not exist yet" % filename
    # device object
    d = GDS()
    # configure
    configure_demo(d, testsignal=True)
    # set configuration
    d.SetConfiguration()
    # get data
    data = d.GetData(d.SamplingRate)
    # save
    np.save(filename, data)
    del data
    # load
    dfromfile = np.load(filename)
    os.remove(filename)
    # show loaded
    scope = Scope(1/d.SamplingRate, modal=True,
                  xlabel="t/s", title='Channel 1')
    scope(dfromfile[:, 0:1])
    plt.show()
    # close
    d.Close()
    del d


def demo_di():
    '''
    This demo:

    - records the DI channel
    - displays it with the live scope

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # configure
    configure_demo(d, testsignal=True, acquire=0)
    d.Trigger = 1
    d.Channels[0].Acquire = 1  # at least one channel needs to be there
    d.SetConfiguration()
    di1 = d.IndexAfter('DI')-1
    di2 = d.IndexAfter('Trigger')-1
    assert di1 == di2
    print('DI channel is ', di1)
    # initialize scope object
    scope = Scope(1/d.SamplingRate, subplots={0: 0, di1: 1}, xlabel=(
        '', 't/s'), ylabel=(u'V/μV', 'DI'), title=('Ch1', 'DI'))
    # get data to scope
    d.GetData(d.SamplingRate, more=scope)
    # close
    d.Close()
    del d


def demo_scope():
    '''
    This demo:

    - records a test signal
    - displays it in the live scope

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # configure
    configure_demo(d, testsignal=True, acquire=0)
    d.Channels[0].Acquire = 1  # at least one channel needs to be there
    d.SetConfiguration()
    # initialize scope
    scope = Scope(1/d.SamplingRate, xlabel='t/s', ylabel=u'V/μV',
                  title="Internal Signal Channels: %s")
    # get data to scope
    d.GetData(d.SamplingRate, more=scope)
    # close
    d.Close()
    del d


def demo_scope_all():
    '''
    This demo:

    - records a test signal for all channels with maximum sampling rate
    - displays it in the live scope

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # configure
    configure_demo(d, testsignal=True, acquire=1)
    sr = d.GetSupportedSamplingRates()[0]
    d.SamplingRate = max(sr.keys())
    if d.DeviceType == DEVICE_TYPE_GHIAMP:
        for i, ch in enumerate(d.Channels):
            if i >= 40:
                ch.Acquire = 0
    elif d.DeviceType == DEVICE_TYPE_GUSBAMP:
        d.SamplingRate = 1200  # >1200 no internal signal
    d.NumberOfScans = sr[d.SamplingRate]
    d.SetConfiguration()
    # initialize scope
    scope = Scope(1/d.SamplingRate, xlabel='t/s', ylabel=u'V/μV',
                  title="Internal Signal Channels: %s")
    # get data every 1/3 of a second
    cnt = d.SamplingRate//3  # determines how often an update happens
    d.GetData(cnt, more=scope)
    # close
    d.Close()
    del d


def demo_scaling():
    '''
    This demo:
    
    - tests the function GetScaling

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # get scaling
    current_scaling = d.GetScaling()
    print(current_scaling)
    # close
    d.Close()
    del d


def demo_impedance():
    '''
    This demo:
    
    - demonstrates impedance measurement

    Have a device:

    - connected to the PC and
    - switched on
    - for g.Nautilus, Cz must be connected to GND

    '''
    d = GDS()
    # get impedances
    impedances = d.GetImpedanceEx()
    print(impedances[0])
    # close
    d.Close()
    del d


def demo_filter():
    '''
    This demo:
    
    - demonstrates the use of filters

    Have a device:

    - connected to the PC and
    - switched on

    '''
    d = GDS()
    # configure to the second lowest sampling rate
    f_s_2 = sorted(d.GetSupportedSamplingRates()[0].items())[1]
    d.SamplingRate, d.NumberOfScans = f_s_2
    # get all applicable filters
    N = [x for x in d.GetNotchFilters()[0] if x['SamplingRate']
         == d.SamplingRate]
    BP = [x for x in d.GetBandpassFilters()[0] if x['SamplingRate']
          == d.SamplingRate]
    # set the first applicable filter
    for ch in d.Channels:
        ch.Acquire = True
        if N:
            ch.NotchFilterIndex = N[0]['NotchFilterIndex']
        if BP:
            ch.BandpassFilterIndex = BP[0]['BandpassFilterIndex']
    # set configuration on device
    d.SetConfiguration()
    # get and display one second of data
    Scope(1/d.SamplingRate, modal=True)(d.GetData(d.SamplingRate))
    plt.show()
    # You wouldn't do the following. Here it is just to check GetConfiguration() functionality.
    for ch in d.Channels:
        ch.Acquire = False
        ch.NotchFilterIndex = -1
        ch.BandpassFilterIndex = -1
    d.GetConfiguration()
    assert d.Channels[0].Acquire == True
    assert d.Channels[0].NotchFilterIndex != - \
        1 or d.Channels[0].BandpassFilterIndex != -1

    # close
    d.Close()
    del d


def demo_all_api():
    '''
    This demo:
    - calls all wrapped API functions
    - can be used as a regression test

    Have a device:

    - connected to the PC and
    - switched on

    '''
    print("Testing communication with the devices")
    print("======================================")
    print()
    # device object
    d = GDS()
    # configure
    configure_demo(d)
    d.Counter = True
    d.SetConfiguration()
    # print all Configs
    print("Devices:")
    for c in d.Configs:
        print(str(c))
        print()
    print()
    # calc number of channels
    print("Configured number of channels: ", d.N_ch_calc())
    print()
    # available channels
    print("Available Channels: ", d.GetAvailableChannels())
    print()
    # device info string
    print("Device informations:")
    dis = d.GetDeviceInformation()
    for di in dis:
        print(di)
        print()
    print()
    # supported sampling rates
    print("Supported sampling rates: ")
    for sr in d.GetSupportedSamplingRates():
        for x in sr:
            print(str(x))
    print()
    # impedances
    print("Measure impedances: ")
    try:
        imps = d.GetImpedanceEx()
        print(imps)
    except GDSError as e:
        print(e)
    print()
    # impedances (deprecated)
    print("Measure impedances (deprecated): ")
    try:
        imps = d.GetImpedance()
        print(imps)
    except GDSError as e:
        print(e)
    print()
    # filters
    print("Bandpass filters:")
    bps = d.GetBandpassFilters()
    for bp in bps:
        for abp in bp:
            print(str(abp))
    print()
    print("Notch filters:")
    notchs = d.GetNotchFilters()
    for notch in notchs:
        for anotch in notch:
            print(str(anotch))
    print()
    # device specific functions

    def print_scalings(scalings):
        for scal in scalings:
            for i, s in enumerate(scal.Offset):
                print("Ch", i+1, ": ", s, ',', scal.Factor[i])
    if d.DeviceType == DEVICE_TYPE_GUSBAMP:
        print('Get and set Async DigitalIOs')
        asyncios = d.GetAsyncDigitalIOs()
        d.SetAsyncDigitalOutputs(asyncios)
    elif d.DeviceType == DEVICE_TYPE_GHIAMP:
        print("Factory scaling:")
        scalings = d.GetFactoryScaling()
        print_scalings(scalings)
    elif d.DeviceType == DEVICE_TYPE_GNAUTILUS:
        d.ResetScaling()
        print("Available DigitalIOs: ")
        dios = d.GetAvailableDigitalIOs()
        for dio in dios:
            print(str(dio))
        print()
        print("Get and set network channel to 19")
        print(d.GetNetworkChannel())
        d.SetNetworkChannel(19)
        print()
        print("Supported sensitivities: ")
        for s in d.GetSupportedSensitivities():
            print(s)
        print()
        print("Supported network channels: ")
        for n in d.GetSupportedNetworkChannels():
            print(n)
        print()
        print("Supported input sources: ")
        for i in d.GetSupportedInputSources()[0]:
            print(i, ' = ', GNAUTILUS_INPUT_SIGNAL.elements[i])
        print()
        print("Get channel names: ")
        for cn in d.GetChannelNames():
            print(cn)
    #calibrate, scaling
    print()
    print("Calibrate...")
    d.Calibrate()
    print("Scaling:")
    scalings = d.GetScaling()
    print_scalings(scalings)
    print("Setting scaling:")
    d.SetScaling(scalings)
    print()
    # data info
    print("Get data info for 500 scans: ")
    dtainfo = d.GetDataInfo(500)
    print(dtainfo)
    print()
    # data
    print("Reading 500 scans: ")
    dta = d.GetData(500)
    print(dta)
    import matplotlib.pyplot as plt
    plt.plot(dta)
    plt.show()
    print()
    # temperature
    print("Temperatures: ")
    # close
    d.Close()
    del d


def demo_usbamp_sync():
    '''
    This demo:

    - configures two g.USBamp with the sinus test signal
    - records all 32 channels of the two synchronized g.USBamp and
    - displays all 32 channels in the time scope.

    Have two switched on g.USBamp devices:

    - connected to the PC and
    - connected with each other via the synch cables

    '''
    dev_names = [n for n, t, u in ConnectedDevices() if t ==
                 DEVICE_TYPE_GUSBAMP and not u]
    devices = ','.join(dev_names)
    print('master,slave = ', devices)
    print()
    if len(dev_names) == 2:
        d = GDS(devices)
        # configure each
        for c in d.Configs:
            c.SamplingRate = 256
            c.NumberOfScans = 8
            c.CommonGround = [0]*4
            c.CommonReference = [0]*4
            c.ShortCutEnabled = 0
            c.CounterEnabled = 0
            c.TriggerEnabled = 0
            c.InternalSignalGenerator.Enabled = 1
            c.InternalSignalGenerator.Frequency = 10
            c.InternalSignalGenerator.WaveShape = GUSBAMP_WAVESHAPE_SINE
            c.InternalSignalGenerator.Amplitude = 200
            c.InternalSignalGenerator.Offset = 0
            for ch in c.Channels:
                ch.Acquire = 1
                # do not use filters
                ch.BandpassFilterIndex = -1
                ch.NotchFilterIndex = -1
                # do not use a bipolar channel
                ch.BipolarChannel = 0
        d.SetConfiguration()
        # create time scope
        scope = Scope(1/c.SamplingRate, xlabel='t/s',
                      ylabel=u'V/μV', title="%s Channels")
        # make scope see 1 second
        d.GetData(1*c.SamplingRate, more=scope)
        # close
        d.Close()
        del d


def demo_remote():
    '''
    This demo:
    
    - shows how to connect a remote PC.

    Have a device:

    - connected to the PC and
    - switched on

    '''
    ip = _this_ip('10.255.255.255')  # use this IP for the test
    print("connecting "+ip)
    for ch in 'U H N'.split():
        try:
            # provide server_ip when connecting remote PC
            d = GDS(gds_device=ch, server_ip=ip)
            print(d.GetDeviceInformation()[0])
            d.Close()
            del d
        except:
            pass


def _run_and_print_demo(demo):
    print()
    print("="*len(demo.__name__))
    print(demo.__name__)
    print("="*len(demo.__name__))
    print()
    print(demo.__doc__)
    print()
    try:
        demo()
        print("Result: Success")
    except Exception as e:
        print("Result: Failed")
        print(e)


def demo_all():
    '''
    Runs all demos.

    '''

    for demo in [demo_all_api, demo_counter, demo_save, demo_di, demo_scope, demo_remote,
                 demo_scope_all, demo_scaling, demo_impedance, demo_filter, demo_usbamp_sync]:
        _run_and_print_demo(demo)


if hasattr(sys, '_called_from_test'):
    import pytest

    @pytest.fixture(params=['g.USBamp', 'g.HIamp', 'g.Nautilus'])
    def devtype(request):
        print()
        print()
        print("Please attach and turn on a " +
              request.param+', then press [ENTER]')
        input()
        print("Running demos for "+request.param +
              ". Close Windows that pop up.")
        return request.param

    def test_all(devtype):
        try:
            from subprocess import run, PIPE
        except:
            from subprocess import check_output, PIPE
            from argparse import Namespace

            def run(*k, **kw):
                del kw['stdout']
                try:
                    stdout = check_output(*k, **kw)
                    returncode = 0
                except:
                    stdout = b"Failed"
                    returncode = 1
                return Namespace(stdout=stdout, returncode=returncode)
        r = run('python -m doctest -v pygds.py', shell=True, stdout=PIPE)
        r1 = r.returncode
        out1 = r.stdout.decode()
        r = run('python pygds.py', shell=True, stdout=PIPE)
        r2 = r.returncode
        out2 = r.stdout.decode()
        with open('pygds_test.log', 'a') as log:
            log.write('='*80+'\n')
            log.write(devtype+'\n')
            log.write('=======\n')
            log.write('doctest\n')
            log.write('=======\n')
            log.write(out1)
            log.write(out2)
        assert r1 == 0 and r2 == 0 and not any(
            'Failed' in x for x in (out1+out2).splitlines())


def main():
    demos = [x for x in dir(sys.modules[__name__]) if x.startswith('demo_')]

    import argparse
    parser = argparse.ArgumentParser(
        description='''pygds.py runs all demo scripts by default.''')
    parser.add_argument('--demo', default=demos[0], const=demos[0], nargs='?',
                        action='store', choices=demos, help="Runs demos. Default is "+demos[0])
    parser.add_argument('--doctest', action='store_true', help="Runs doctests")

    try:
        args = parser.parse_args()
        Initialize()

        if args.doctest:
            import doctest
            doctest.testmod()
        else:
            demo = eval(args.demo)
            _run_and_print_demo(demo)
    except AttributeError:  # because demo not defined in case of --help
        pass


if __name__ == '__main__':
    main()
