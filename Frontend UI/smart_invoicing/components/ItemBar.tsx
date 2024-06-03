import React, { useEffect, useState } from "react";
import { Item } from "@/app/page";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Pencil, X } from "lucide-react";

type Props = {
  items: Item[];
  setitems: React.Dispatch<React.SetStateAction<Item[]>>;
  detection: string;
  price: number;
  BarcodedItem: boolean;
};

export default function ItemBar({
  detection,
  price,
  BarcodedItem,
  items,
  setitems,
}: Props) {
  const [value, setValue] = useState(0);
  const [edititemname, setEdititemname] = useState("");
  const [edititemprice, setEdititemprice] = useState(0);
  const [editBarcodedItem, setBarcodedItem] = useState(false);

  function deleteItem(index: number) {
    const updatedItems = [...items];
    updatedItems.splice(index, 1);
    setitems(updatedItems);
  }
  function editItem(index: number) {
    const updatedItems = [...items];
    updatedItems[index] = {
      detection: edititemname,
      price: edititemprice,
      BarcodedItem: editBarcodedItem,
      value: value,
    };
    setitems(updatedItems);
  }

  useEffect(() => {
    if (value) {
      const updatedItems = [...items];
      const index = items.findIndex((item) => item.detection === detection);

      updatedItems[index] = {
        detection,
        price,
        BarcodedItem,
        value: value,
      };

      setitems(updatedItems);
    }
  }, [value]);
  return (
    <>
      <div className="flex flex-row justify-between w-full p-2 mt-4">
        <div className="flex flex-col">
          <div className="text-lg font-bold">{detection}</div>
          <div className="text-sm">Rs.{price}/kg</div>
        </div>
        <div className="flex flex-col">
          <div className="text-lg font-bold">
            {BarcodedItem ? "Quantity" : "Weight(kg)"}
          </div>
          <div className="text-sm">
            <input
              type="number"
              value={value}
              pattern="[0-9]*"
              className="bg-gray-200 rounded-md p-1 w-1/2 "
              onChange={(e) => setValue(parseFloat(e.target.value))}
            />
          </div>
        </div>
        {/* total price */}
        <div className="flex flex-col">
          <div className="text-lg font-bold">Total</div>
          <div className="text-sm ">Rs.{value * price || "0"}</div>
        </div>
        {/* remove item */}
        <div className="flex flex-col-reverse   items-end gap-5">
          <Dialog>
            <DialogTrigger>
              <div
                onClick={() => {}}
                className="text-sm font-bold px-2 gap-3 py-1 hover:bg-black hover:text-white bg-gray-200 rounded-full  flex items-center justify-center cursor-pointer transition-all duration-150"
              >
                <span>Edit</span>
                <Pencil size={16} />
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <div className="flex flex-col gap-3  mt-5  ">
                  <div className="flex justify-between items-center">
                    <label htmlFor="detection">Name</label>
                    <input
                      type="text"
                      placeholder="item"
                      className="bg-gray-200 rounded-md p-2"
                      onChange={(e) => {
                        setEdititemname(e.target.value);
                      }}
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <label htmlFor="price">Price</label>
                    <input
                      type="text"
                      placeholder="price"
                      className="bg-gray-200 rounded-md p-2"
                      onChange={(e) => {
                        setEdititemprice(parseFloat(e.target.value));
                      }}
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <label htmlFor="BarcodedItem">Barcoded Item</label>
                    <input
                      type="checkbox"
                      className="bg-gray-200 rounded-md p-2"
                      onChange={(e) => {
                        setBarcodedItem(e.target.checked);
                      }}
                    />
                  </div>

                  <div className="flex w-full justify-end items-center">
                    <div
                      onClick={() => {
                        editItem(
                          items.findIndex(
                            (item) => item.detection === detection
                          )
                        );
                      }}
                      className="border-2 border-blue-500 rounded-full px-2 text-base text-blue-500 hover:bg-blue-500 hover:text-white transition-all duration-150"
                    >
                      Edit
                    </div>
                  </div>
                </div>
              </DialogHeader>
            </DialogContent>
          </Dialog>
          <div
            onClick={() => {
              deleteItem(
                items.findIndex((item) => item.detection === detection)
              );
            }}
            className=""
          >
            <X />
          </div>
        </div>
      </div>
    </>
  );
}
