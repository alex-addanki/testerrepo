import PIL
from PIL import Image


def creategauge(value,filename):
    percent = int(value)  # Percent for gauge
    output_file_name = filename

    # X and Y coordinates of the center bottom of the needle starting from the top left corner
    #   of the image
    x = 825
    y = 825
    loc = (x, y)

    percent = percent / 100
    rotation = 180 * percent  # 180 degrees because the gauge is half a circle
    rotation = 90 - rotation  # Factor in the needle graphic pointing to 50 (90 degrees)

    dial = Image.open('images/needle.png')
    dial = dial.rotate(rotation, resample=PIL.Image.BICUBIC, center=loc)  # Rotate needle

    gauge = Image.open('images/gauge.png')
    gauge.paste(dial, mask=dial)  # Paste needle onto gauge
    try:
        gauge.save('images/NEW_gauge.png')
        image = Image.open('images/NEW_gauge.png')
        new_image = image.resize((75, 75))
        new_image.save(output_file_name)
        ###new_image.show()
        value = 0
    except:
        value =  1
    return value
