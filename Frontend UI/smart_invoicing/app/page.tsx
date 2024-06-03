"use client";

import camFrame from "C:/projects/Smart_Invoicing/Smart_Invoicing_system/code.jpg";
import { usePDF } from "react-to-pdf";

import { use, useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import ItemBar from "../components/ItemBar";
import Image from "next/image";
import { Plus, X } from "lucide-react";

export type Item = {
  detection: string;
  price: number;
  BarcodedItem: boolean;
  value: number;
};

//delete item from the list

export default function Home() {
  const [items, setitems] = useState<Item[]>([]);
  const [loading, setloading] = useState(false);
  const [weight, setweight] = useState("exp12");
  const [source, setsource] = useState("0");
  const [confThres, setconfThres] = useState(0.85);
  const [totalcartvalue, settotalcartvalue] = useState(0);
  const [modal, setmodal] = useState(false);
  const [additemname, setAdditemname] = useState("");
  const [additemprice, setAdditemprice] = useState(0);
  const [IsbarcodedItem, setIsbarcodedItem] = useState(false);
  const [Bill, setBill] = useState(0);
  const [time, settime] = useState("");

  const [htmlData, sethtmlData] = useState("");

  const { toPDF, targetRef } = usePDF({ filename: `${Bill}.pdf` });

  function printInvoice() {
    //save the invoice as pdf and name it as billno
  }

  async function detect() {
    const response = await fetch("http://localhost:5000/detect");
    const data = await response.json();

    // add item to the list
    setitems([...items, data]);
    console.log(data);
  }

  async function saveInvoiceDetails() {
    const BillNo = Math.floor(Math.random() * 1000000) * Date.now();
    const date = new Date().toLocaleDateString();
    setBill(BillNo);
    settime(date);
    try {
      const body = {
        items: items,
        totalcartvalue: totalcartvalue,
        BillNo: BillNo,
        date: date,
      };

      const response = await fetch("http://localhost:5000/save", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();
      console.log(data, body);
    } catch (error) {
      console.log(error);
    }
  }

  async function start() {
    const body = {
      weights: `runs/train/${weight}/weights/best.pt`,
      source: source,
      conf_thres: confThres,
    };
    console.log(body);
    //post request to start the detection
    setloading(true);

    try {
      const response = await fetch("http://localhost:5000/opencam", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });
      console.log("running");
    } catch (error) {
      console.log(error);
    }
  }

  async function stop() {
    // post request to stop the detection
    setloading(false);
    setitems([]);
    settotalcartvalue(0);
    try {
      const response = await fetch("http://localhost:5000/stop", {
        method: "POST",
      });
    } catch (error) {
      console.log(error);
    }
  }

  async function QRgen() {
    try {
      const response = await fetch("http://localhost:5000/QRgen", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          total_price: totalcartvalue.toString(),
        }),
      });

      //this return <svg>...<.svg> as string
      const data = await response.text();

      sethtmlData(data);
    } catch (error) {
      console.log(error);
    }
  }

  useEffect(() => {
    settotalcartvalue(
      items.reduce((acc, item) => acc + item.price * item.value, 0)
    );
  }, [items]);

  return (
    <>
      <div className="flex h-screen w-full">
        <div className="flex flex-col  h-full w-3/6">
          <div className="w-full  h-full justify-between flex flex-col ">
            <div className="h-4/6 flex justify-center items-center border-b-2 p-4 border-black">
              <Image
                src={camFrame}
                width={500}
                height={500}
                alt="camera frame"
              />
            </div>
            <div className="w-full flex  flex-col p-4 rounded-md bg-white ">
              <div className="p-3 w-full">
                <button
                  onClick={detect}
                  className={` w-full hover:bg-black hover:text-white transition-all duration-150 border-2 border-black px-4 py-2 italic `}
                >
                  DETECT
                </button>
              </div>
              <div className="flex justify-between items-center p-3 gap-2">
                <div className="flex flex-col gap-2">
                  <label htmlFor="weight">weight</label>
                  <input
                    type="text"
                    placeholder={weight}
                    className="bg-gray-200 rounded-md p-2 "
                    onChange={(e) => setweight(e.target.value)}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <label htmlFor="source">source</label>
                  <input
                    type="text"
                    placeholder={source}
                    className="bg-gray-200 rounded-md p-2"
                    onChange={(e) => setsource(e.target.value)}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <label htmlFor="confThres">confThres</label>
                  <input
                    type="text"
                    placeholder={confThres.toString()}
                    className="bg-gray-200 rounded-md p-2"
                    onChange={(e) => setconfThres(parseInt(e.target.value))}
                  />
                </div>
              </div>
              <div className="flex justify-between p-3">
                <button
                  onClick={start}
                  className={`${
                    loading && "bg-black text-white"
                  } hover:bg-black hover:text-white transition-all duration-150 border-2 border-black px-4 py-2 italic`}
                >
                  {loading ? (
                    <>
                      <div className="flex gap-2">
                        <span className="animate-pulse">RUNNING</span>
                      </div>
                    </>
                  ) : (
                    "START"
                  )}
                </button>
                <button
                  onClick={stop}
                  className="hover:bg-black hover:text-white transition-all duration-150 border-2 border-black px-4 py-2 italic"
                >
                  STOP
                </button>
              </div>
            </div>
          </div>
        </div>
        <div className="flex justify-between p-3  flex-col border-black border-l-2 h-full w-full">
          <div className="w-full p-4 rounded-md shadow-[0_3px_10px_rgb(0,0,0,0.2)] overflow-y-scroll bg-white">
            <div className="w-full flex justify-end ">
              <Dialog>
                <DialogTrigger>
                  <div className="flex gap-1 items-center border-2 border-blue-500 rounded-full px-2 text-xl text-blue-500 hover:bg-blue-500 hover:text-white transition-all duration-150">
                    <Plus /> Add Item
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
                            setAdditemname(e.target.value);
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
                            setAdditemprice(parseFloat(e.target.value));
                          }}
                        />
                      </div>
                      <div className="flex justify-between items-center">
                        <label htmlFor="barcodedItem">Barcoded Item</label>
                        <input
                          type="checkbox"
                          id="barcodedItem"
                          checked={IsbarcodedItem}
                          onChange={(e) => setIsbarcodedItem(e.target.checked)}
                        />
                      </div>

                      <div className="flex w-full justify-end items-center">
                        <div
                          onClick={() => {
                            setitems([
                              ...items,
                              {
                                detection: additemname,
                                price: additemprice,
                                BarcodedItem: IsbarcodedItem,
                                value: 0,
                              },
                            ]);
                          }}
                          className="border-2 border-blue-500 rounded-full px-2 text-base text-blue-500 hover:bg-blue-500 hover:text-white transition-all duration-150"
                        >
                          ADD
                        </div>
                      </div>
                    </div>
                  </DialogHeader>
                </DialogContent>
              </Dialog>
            </div>
            {items.length > 0 ? (
              <>
                {items.map((item, index) => (
                  <div key={index} className="flex ">
                    <ItemBar
                      detection={item.detection}
                      price={item.price}
                      BarcodedItem={item.BarcodedItem}
                      items={items}
                      setitems={setitems}
                    />
                  </div>
                ))}
              </>
            ) : (
              <>
                <span className="flex justify-center text-2xl">No items</span>
              </>
            )}
          </div>
          <div className="w-full bg-white flex justify-between p-4 rounded-md shadow-[0_3px_10px_rgb(0,0,0,0.2)]">
            <div className="text-2xl">Total amount:</div>
            <div className="text-2xl">Rs {totalcartvalue}</div>
            <Dialog>
              <DialogTrigger>
                <div
                  onClick={() => {
                    saveInvoiceDetails();
                    QRgen();
                  }}
                  className="border-blue-500 rounded-full text-2xl border-2 px-2 font-semibold text-blue-500 cursor-pointer hover:bg-blue-500 hover:text-white transition-all duration-150"
                >
                  PAY
                </div>
              </DialogTrigger>
              <DialogContent id="invoice">
                <DialogHeader>
                  <DialogDescription
                    ref={targetRef}
                    className="flex w-full text-black"
                  >
                    <div className="flex w-full flex-col gap-2">
                      <div className="flex flex-col  items-center mb-10">
                        <span className="text-5xl">INVOICE</span>
                        <span className="text-3xl">Smart Shop</span>
                      </div>
                      <div className="">Bill No: {Bill}</div>
                      <div className="">Date: {time}</div>

                      <table className="text-lg mt-5 ">
                        <thead>
                          <tr>
                            <th>Item Name</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th className="text-right">Total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {items.map((item, index) => (
                            <tr key={index}>
                              <td>{item.detection}</td>
                              <td>{item.value}</td>
                              <td>₹{item.price}</td>
                              <td className="text-right">
                                ₹{item.value * item.price}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <hr className="border border-black mt-16" />
                      <div className="flex  justify-between text-2xl ">
                        <div className="">Total:</div>
                        <div className=""> ₹ {totalcartvalue}</div>
                      </div>
                      <div className="flex w-full justify-center items-center ">
                        {htmlData ? (
                          <div
                            className="w-1/2 h-full flex justify-center items-center "
                            dangerouslySetInnerHTML={{ __html: htmlData }}
                          ></div>
                        ) : (
                          <>
                            <div className="h-5 w-5 border-2 border-t-0 border-blue-500 rounded-full animate-spin"></div>
                          </>
                        )}
                      </div>

                      <div className="flex justify-center items-center">
                        <div className=" italic">PAY SAUBHAGYA PRASAD</div>
                      </div>
                      <div className="flex justify-center items-center">
                        <div className="">
                          **THANK YOU FOR SHOPING WITH US**
                        </div>
                      </div>
                    </div>
                  </DialogDescription>
                  <div className="flex justify-end mt-10">
                    <button
                      // onClick={() => {
                      //   printInvoice();
                      // }}
                      onClick={() => toPDF()}
                      className="border-2 border-blue-500 rounded-md px-2 text-lg text-blue-500 hover:bg-blue-500 hover:text-white transition-all duration-150"
                    >
                      Print Invoice
                    </button>
                  </div>
                </DialogHeader>
                <DialogFooter> </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>
    </>
  );
}
