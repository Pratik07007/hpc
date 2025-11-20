#include <stdio.h>
#include <string.h>
#include <pthread.h>

typedef struct
{
    int id;
    const char *name;
    double price;
    int taxable;
} Product;

typedef struct
{
    const Product *product;
    int qty;
} OrderItem;

typedef struct
{
    double base;
    double rate;
    double *out;
} PercentArg;

static void *compute_percentage(void *arg)
{
    PercentArg *a = (PercentArg *)arg;
    *(a->out) = a->base * a->rate / 100.0;
    return NULL;
}

static Product PRODUCTS[] = {
    {1, "Rice", 120.0, 0},
    {2, "Milk", 150.0, 0},
    {3, "Eggs", 1400.0, 0},
    {4, "Chocolate", 90.0, 1},
    {5, "Samsung", 30000.0, 1},
    {6, "Pan", 150.0, 0},
    {7, "Sugar", 80.0, 0},
    {8, "Apples", 110.0, 1},
    {9, "Avacados", 400.0, 1},
    {10, "Pen", 50.0, 0}};

static int PRODUCT_COUNT = sizeof(PRODUCTS) / sizeof(PRODUCTS[0]);

static const Product *find_product_by_id(int id)
{
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        if (PRODUCTS[i].id == id)
            return &PRODUCTS[i];
    }
    return NULL;
}

static void print_products()
{
    printf("Products List\n");
    printf("%-3s %-12s %-10s %-7s\n", "ID", "Name", "Price", "Taxable");
    for (int i = 0; i < PRODUCT_COUNT; i++)
    {
        printf("%-3d %-12s %-10.2f %-7d\n", PRODUCTS[i].id, PRODUCTS[i].name, PRODUCTS[i].price, PRODUCTS[i].taxable);
    }
}

int main()
{
    OrderItem items[100];
    int item_count = 0;
    int invoice_no = 1;
    double tax_rate = 0.0;
    double discount_rate = 0.0;

    print_products();

    while (1)
    {
        int pid;
        int qty;
        char cont;
        printf("\nEnter Product ID: ");
        if (scanf("%d", &pid) != 1)
            return 0;
        const Product *p = find_product_by_id(pid);
        if (!p)
        {
            printf("Invalid Product ID\n");
            continue;
        }
        printf("Enter Quantity: ");
        if (scanf("%d", &qty) != 1 || qty <= 0)
        {
            printf("Invalid Quantity\n");
            continue;
        }
        items[item_count].product = p;
        items[item_count].qty = qty;
        item_count++;
        printf("Do you want to place another order? (y/n): ");
        scanf(" %c", &cont);
        if (cont == 'n' || cont == 'N')
            break;
    }

    printf("Enter discount percentage (0-99): ");
    scanf("%lf", &discount_rate);
    if (discount_rate < 0)
        discount_rate = 0;
    if (discount_rate > 99)
        discount_rate = 99;
    printf("Enter tax rate in percentage: ");
    scanf("%lf", &tax_rate);

    printf("\nInvoice No: %d\n", invoice_no);
    printf("%-3s %-15s %-8s %-10s %-10s\n", "SC", "Product Name", "Quantity", "Rate", "Total");

    double subtotal = 0.0;
    double taxable_base = 0.0;
    for (int i = 0; i < item_count; i++)
    {
        const Product *p = items[i].product;
        int q = items[i].qty;
        double line_total = p->price * q;
        subtotal += line_total;
        if (p->taxable)
            taxable_base += line_total;
        printf("%-3d %-15s %-8d %-10.2f %-10.2f\n", i + 1, p->name, q, p->price, line_total);
    }

    double discount_amount = 0.0;
    double tax_amount = 0.0;

    pthread_t t1, t2;
    PercentArg a1 = {subtotal, discount_rate, &discount_amount};
    PercentArg a2 = {taxable_base, tax_rate, &tax_amount};
    pthread_create(&t1, NULL, compute_percentage, &a1);
    pthread_create(&t2, NULL, compute_percentage, &a2);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    double discounted_subtotal = subtotal - discount_amount;
    double grand_total = discounted_subtotal + tax_amount;

    printf("\nSubtotal: %.2f\n", subtotal);
    printf("Discount (%.0f%%): %.2f\n", discount_rate, discount_amount);
    printf("Discounted Subtotal: %.2f\n", discounted_subtotal);
    printf("Taxable Amount: %.2f\n", taxable_base);
    printf("Tax (%.0f%%): %.2f\n", tax_rate, tax_amount);
    printf("Grand Total: %.2f\n", grand_total);
    return 0;
}