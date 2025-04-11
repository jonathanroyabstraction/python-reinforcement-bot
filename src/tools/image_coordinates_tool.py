import cv2
import argparse
import numpy as np

def get_coordinates(image_path):
    """Display image and let user get coordinates by clicking."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Store click coordinates
    coords = []
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click
            coords.append((x, y))
            # Draw a point at click location
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"({x},{y})", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Image", image)
            print(f"Clicked at: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move
            # Show coordinates in window title
            cv2.setWindowTitle("Image", f"Image - Mouse at: ({x}, {y})")
    
    # Display the image
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    
    print("Click on the image to record coordinates (press ESC to exit)")
    print("Top-left and bottom-right clicks will define a bounding box")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    
    # If we have at least 2 points, calculate bounding box
    if len(coords) >= 2:
        x1, y1 = coords[0]
        x2, y2 = coords[-1]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        x = min(x1, x2)
        y = min(y1, y2)
        print(f"\nBoundingBox(x={x}, y={y}, width={width}, height={height})")
    
    return coords

def main():
    """Get image coordinates."""
    parser = argparse.ArgumentParser(description="Get image coordinates")
    parser.add_argument("--image_path", type=str, help="Path to screenshot image")
    
    args = parser.parse_args()
    
    if (args.image_path):
        get_coordinates(args.image_path)
    else:
        print("No image path has been provided. Quitting.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()