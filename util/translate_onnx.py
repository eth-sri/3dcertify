import numpy as np
import onnx
from onnx import numpy_helper
from onnx.numpy_helper import to_array, from_array


def expand_conv_node(node: onnx.NodeProto) -> onnx.NodeProto:
    attributes = {}
    for attribute in node.attribute:
        if attribute.name == "dilations":
            dilations = attribute.ints
            dilations.insert(1, 1)
            attributes['dilations'] = dilations
        elif attribute.name == "group":
            group = attribute.i
            attributes['group'] = group
        elif attribute.name == "kernel_shape":
            kernel_shape = attribute.ints
            kernel_shape.insert(1, 1)
            attributes['kernel_shape'] = kernel_shape
        elif attribute.name == "pads":
            pads = attribute.ints
            pads.insert(1, 0)
            pads.insert(3, 0)
            attributes['pads'] = pads
        elif attribute.name == "strides":
            strides = attribute.ints
            strides.insert(1, 1)
            attributes['strides'] = strides
        else:
            assert False, "Unsupported Attribute in MaxPool layer: {}".format(attribute)

    return onnx.helper.make_node(node.op_type, node.input, node.output, **attributes)


def expand_pool(node: onnx.NodeProto) -> onnx.NodeProto:
    attributes = {}
    for attribute in node.attribute:
        if attribute.name == "kernel_shape":
            kernel_shape = attribute.ints
            kernel_shape.insert(1, 1)
            attributes['kernel_shape'] = kernel_shape
        elif attribute.name == "pads":
            pads = attribute.ints
            pads.insert(1, 0)
            pads.insert(3, 0)
            attributes['pads'] = pads
        elif attribute.name == "strides":
            strides = attribute.ints
            strides.insert(1, 1)
            attributes['strides'] = strides
        else:
            assert False, "Unsupported Attribute in pooling layer: {}".format(attribute)

    return onnx.helper.make_node(node.op_type, node.input, node.output, **attributes)


def replace_tensor(name: str, axis: int, graph: onnx.GraphProto):
    for i, tensor in enumerate(graph.initializer):
        if tensor.name == name:
            new_tensor = expand_tensor(tensor, axis)
            graph.initializer.remove(tensor)
            graph.initializer.insert(i, new_tensor)
    for i, input in enumerate(graph.input):
        if input.name == name:
            new_input = expand_input(input, axis)
            graph.input.remove(input)
            graph.input.insert(i, new_input)


def expand_tensor(tensor: onnx.TensorProto, axis: int) -> onnx.TensorProto:
    dims = tensor.dims
    dims.insert(axis, 1)
    return onnx.helper.make_tensor(tensor.name, tensor.data_type, dims, tensor.raw_data, raw=True)


def expand_input(input: onnx.ValueInfoProto, axis: int) -> onnx.ValueInfoProto:
    shape = [i.dim_value for i in input.type.tensor_type.shape.dim]
    shape.insert(axis, 1)
    return onnx.helper.make_tensor_value_info(input.name, input.type.tensor_type.elem_type, shape)


def find_tensor(id: str, graph: onnx.GraphProto) -> onnx.TensorProto:
    for tensor in graph.initializer:
        if tensor.name == id:
            return tensor
    assert False, f"No tensor defined with id {id}"


def merge_batch_norm(affine_node: onnx.NodeProto, bn_node: onnx.NodeProto, graph: onnx.GraphProto):
    bn_scale = to_array(find_tensor(bn_node.input[1], graph))
    bn_shift = to_array(find_tensor(bn_node.input[2], graph))
    bn_mean = to_array(find_tensor(bn_node.input[3], graph))
    bn_var = to_array(find_tensor(bn_node.input[4], graph))
    eps = 1E-05
    for attribute in bn_node.attribute:
        if attribute.name == "epsilon":
            eps = attribute.f
    for i, tensor in enumerate(graph.initializer):
        # weight tensor
        if tensor.name == affine_node.input[1]:
            conv_weight = to_array(tensor)
            weight_size = conv_weight.shape
            out_features = weight_size[0]
            conv_weight = np.reshape(conv_weight, (out_features, -1))
            bn_weight = np.diag(bn_scale / np.sqrt(bn_var + eps))
            # bn_weight = np.diag(bn_scale)
            fused_weight = np.matmul(bn_weight, conv_weight)
            fused_weight = np.reshape(fused_weight, weight_size)
            fused_weight = from_array(fused_weight, name=tensor.name)
            graph.initializer.remove(tensor)
            graph.initializer.insert(i, fused_weight)
        # bias tensor
        elif tensor.name == affine_node.input[2]:
            conv_bias = to_array(tensor)
            fused_bias = bn_shift + (conv_bias - bn_mean) * bn_scale / np.sqrt(bn_var + eps)
            fused_bias = from_array(fused_bias, name=tensor.name)
            graph.initializer.remove(tensor)
            graph.initializer.insert(i, fused_bias)


def cleanup_unused_initializers(model: onnx.ModelProto):
    graph = model.graph
    to_remove = []
    for tensor in graph.initializer:
        used = False
        for node in graph.node:
            for input in node.input:
                if input == tensor.name:
                    used = True
        if not used:
            to_remove.append(tensor)
    for tensor in to_remove:
        graph.initializer.remove(tensor)


def remove_empty_paddings(model: onnx.ModelProto):
    graph = model.graph
    for i, node in enumerate(graph.node):
        if node.op_type == "Pad":
            graph.node[i + 1].input[0] = node.input[0]
            graph.node.remove(node)


def merge_bn_layers(model: onnx.ModelProto):
    graph = model.graph
    to_remove = []
    for i, node in enumerate(graph.node):
        if node.op_type == "BatchNormalization" and graph.node[i - 1].op_type in ["Conv", "Gemm"]:
            affine_node = graph.node[i - 1]
            assert affine_node.output[0] == node.input[0], "Consecutive Conv and BN nodes are not connected"
            merge_batch_norm(affine_node, node, graph)
            affine_node.output[0] = node.output[0]
            to_remove.append(node)
    #        graph.node.remove(node)
    for node in to_remove:
        graph.node.remove(node)


def expand_1d_operations(model: onnx.ModelProto):
    graph = model.graph
    for i, node in enumerate(graph.node):
        if node.op_type == "Conv":
            new_node = expand_conv_node(node)
            graph.node.remove(node)
            graph.node.insert(i, new_node)
            replace_tensor(new_node.input[0], 3, graph)
            replace_tensor(new_node.input[1], 3, graph)
        elif node.op_type in ["MaxPool", "AveragePool"]:
            new_node = expand_pool(node)
            graph.node.remove(node)
            graph.node.insert(i, new_node)
            replace_tensor(new_node.input[0], 3, graph)
        elif node.op_type == "Constant":
            if graph.node[i + 1].op_type == "Tile":
                attribute = node.attribute[0]
                tensor = attribute.t
                new_value = np.append(numpy_helper.to_array(tensor), 1)
                new_tensor = onnx.helper.make_tensor(tensor.name, tensor.data_type, (4,), new_value, raw=False)
                new_attribute = onnx.helper.make_attribute(attribute.name, new_tensor)
                node.attribute.remove(attribute)
                node.attribute.insert(0, new_attribute)


def remove_first_affine(model: onnx.ModelProto):
    graph = model.graph
    first_affine = graph.node[0]
    identity_weight = from_array(np.expand_dims(np.eye(64, dtype=np.float32), axis=(2, 3)), 'identity_weight')
    identity_bias = from_array(np.zeros(64, dtype=np.float32), 'identity_bias')
    identity_node = onnx.helper.make_node(
        op_type='Conv',
        inputs=[first_affine.input[0], identity_weight.name, identity_bias.name],
        outputs=first_affine.output,
        name=first_affine.name,
        kernel_shape=[1, 1],
        dilations=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )
    graph.node.remove(first_affine)
    graph.initializer.insert(0, identity_bias)
    graph.initializer.insert(0, identity_weight)
    graph.node.insert(0, identity_node)

    input = graph.input[0]
    shape = [i.dim_value for i in input.type.tensor_type.shape.dim]
    shape[1] = 64
    new_input = onnx.helper.make_tensor_value_info(input.name, input.type.tensor_type.elem_type, shape)
    graph.input.remove(input)
    graph.input.insert(0, new_input)


def translate(model: onnx.ModelProto, remove_first_layer: bool):
    merge_bn_layers(model)
    expand_1d_operations(model)
    remove_empty_paddings(model)
    if remove_first_layer:
        remove_first_affine(model)
    cleanup_unused_initializers(model)
    onnx.checker.check_model(model)


if __name__ == "__main__":
    model = onnx.load_model("out/face_propagation_test/cls_model_249.onnx")

    print(onnx.helper.printable_graph(model.graph))
    translate(model, False)
    print(onnx.helper.printable_graph(model.graph))

    dummy_input = np.random.randn(1, 3, 1024).astype(np.float32)
    import onnxruntime

    session = onnxruntime.InferenceSession("out/face_propagation_test/cls_model_249.onnx")
    inputs = {session.get_inputs()[0].name: dummy_input.copy()}
    original_outputs = session.run(None, inputs)

    session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = {session.get_inputs()[0].name: np.expand_dims(dummy_input.copy(), axis=-1)}
    transformed_outputs = session.run(None, inputs)

    print("max abs difference: ", np.max(np.abs(np.array(original_outputs) - np.array(transformed_outputs))))
    print("mean abs difference:", np.mean(np.abs(np.array(original_outputs) - np.array(transformed_outputs))))
    np.testing.assert_allclose(np.array(original_outputs), np.array(transformed_outputs), rtol=1e-03, atol=1e-05)
