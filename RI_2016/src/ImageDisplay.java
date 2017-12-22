import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.JFrame;

public class ImageDisplay extends Panel {
  BufferedImage  image;
  public ImageDisplay(BufferedImage  im) {
  this.image = im;
  }

  public void paint(Graphics g) {
  g.drawImage( image, 0, 0, null);
  }
}
