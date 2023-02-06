
def first_channels(input_channels, data, **kwargs):

    if data.shape[-1] != input_channels:
        print(f'data channels: {data.shape[-1]}, args channels: {input_channels}')
    if input_channels < 1:
        print('args channels less than 1, using data channels')
        input_channels = data.shape[-1]
    elif data.shape[-1] < input_channels:
        print('args channels more than data channels, using data channels')
        input_channels = data.shape[-1]
    elif data.shape[-1] > input_channels:
        print(f'extracting first {input_channels} channels from data')

    return data[..., 0:input_channels]
