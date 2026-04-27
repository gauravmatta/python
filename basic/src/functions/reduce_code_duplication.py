def print_order(name,tea_type):
    print(f"{name} ordered tea type: {tea_type} U@")

def fetch_sales():
    print("Fetching the sales data")

def filter_valid_sales():
    print("Filtering the valid sales data")

def summarize_valid_sales():
    print("Summarizing the valid sales data")

def generate_report():
    print("Generating the report data")
    fetch_sales()
    filter_valid_sales()
    summarize_valid_sales()
    print("Report generated successfully")

print_order("Ravi",tea_type="Masala")
print_order("Prince",tea_type="Ginger")
print_order("Gaurav","Tulsi")
generate_report()