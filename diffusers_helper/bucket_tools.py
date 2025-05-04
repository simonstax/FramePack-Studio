bucket_options = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    # Always use the 640 resolution bucket sizes
    resolution = 640
    min_diff = float('inf')
    best_bucket = None
    
    # Find the bucket size where the first parameter (width) is closest to the slider value
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        diff = abs(bucket_w - resolution)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

