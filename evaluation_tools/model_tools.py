# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from BOP Toolkit (https://github.com/thodan/bop_toolkit)
# Copyright (c) Tomas Hodan
# Licensed under the MIT License [see LICENSE_BOP_TOOLKIT in the LICENSES folder for details]
# ------------------------------------------------------------------------

import numpy as np
import struct
import imageio


def load_im(path):
    """Loads an image from a file.
    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    im = imageio.imread(path)
    return im


def load_ply(path):
    """Loads a 3D mesh model from a PLY file.
    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray), optional
     - 'colors' (nx3 ndarray), optional
     - 'faces' (mx3 ndarray), optional
     - 'texture_uv' (nx2 ndarray), optional
     - 'texture_uv_face' (mx6 ndarray), optional
     - 'texture_file' (string), optional
    """
    f = open(path, 'rb')

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:

        # Strip the newline character(s).
        line = f.readline().decode('utf8').rstrip('\n').rstrip('\r')

        if line.startswith('comment TextureFile'):
            texture_file = line.split()[-1]
        elif line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split()
            if elems[-1] == 'vertex_indices' or elems[-1] == 'vertex_index':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            elif elems[-1] == 'texcoord':
                # (name of the property, data type)
                face_props.append(('texcoord', elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(('texcoord_ind_' + str(i), elems[3]))
            else:
                print('Warning: Not supported face property: ' + elems[-1])
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model['texture_file'] = texture_file
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]

    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    is_texture_pt = False
    if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model['texture_uv'] = np.zeros((n_pts, 2), np.float)

    is_texture_face = False
    if {'texcoord'}.issubset(set(face_props_names)):
        is_texture_face = True
        model['texture_uv_face'] = np.zeros((n_faces, 6), np.float)

    # Formats for the binary case.
    formats = {
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                      'red', 'green', 'blue', 'texture_u', 'texture_v']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

        if is_texture_pt:
            model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
            model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        raise ValueError('Only triangular faces are supported.')
                elif prop[0] == 'texcoord':
                    if val != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError('Only triangular faces are supported.')
                elif prop[0] == 'texcoord':
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

        if is_texture_face:
            for i in range(6):
                model['texture_uv_face'][face_id, i] = float(
                    prop_vals['texcoord_ind_{}'.format(i)])

    f.close()

    return model

